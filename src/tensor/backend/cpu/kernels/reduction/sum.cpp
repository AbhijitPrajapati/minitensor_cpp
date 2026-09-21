#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/buffer_access.hpp"
#include "tensor/backend/cpu/kernels/common/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/reduction/sum.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        struct Dimension final
        {
            Shape::size_type extent;
            Layout::stride_type step;
            Layout::offset_type reset;
        };

        void advance(
            std::vector<Shape::size_type> &indices,
            std::span<const Dimension> dimensions,
            Layout::offset_type &offset) noexcept
        {
            for (std::size_t i = dimensions.size(); i > 0; --i)
            {
                const std::size_t dimension_index = i - 1;
                const Dimension &dimension = dimensions[dimension_index];
                Shape::size_type &index = indices[dimension_index];

                ++index;
                if (index < dimension.extent)
                {
                    offset += dimension.step;
                    return;
                }

                index = 0;
                offset -= dimension.reset;
            }
        }

        [[nodiscard]] bool reduces_contiguous_suffix(
            Shape::size_type rank,
            std::span<const Shape::size_type> axes) noexcept
        {
            if (axes.empty())
            {
                return false;
            }

            const Shape::size_type first_axis = rank - axes.size();
            for (Shape::size_type i = 0; i < axes.size(); ++i)
            {
                if (axes[i] != first_axis + i)
                {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] Shape::size_type reduction_size(
            const Shape &shape,
            std::span<const Shape::size_type> axes) noexcept
        {
            Shape::size_type size = 1;
            for (const Shape::size_type axis : axes)
            {
                size *= static_cast<Shape::size_type>(shape[axis]);
            }
            return size;
        }

        template <CpuElement T>
        void sum_contiguous_suffix(
            const TensorView &input,
            MutableTensorView output,
            Shape::size_type values_per_output)
        {
            const T *input_data = data<T>(input);
            T *output_data = data<T>(output);
            const auto input_offset = static_cast<std::size_t>(input.layout().offset());
            const auto output_offset = static_cast<std::size_t>(output.layout().offset());

            for (Shape::size_type output_linear = 0;
                 output_linear < output.shape().numel();
                 ++output_linear)
            {
                const T *values =
                    input_data + input_offset + output_linear * values_per_output;
                T accumulator{};
                for (Shape::size_type i = 0; i < values_per_output; ++i)
                {
                    accumulator += values[i];
                }
                output_data[output_offset + output_linear] = accumulator;
            }
        }

        template <CpuElement T>
        void sum_strided(
            const TensorView &input,
            MutableTensorView output,
            std::span<const Shape::size_type> axes,
            Shape::size_type values_per_output)
        {
            const Shape &input_shape = input.shape();
            const Layout &input_layout = input.layout();
            std::vector<bool> is_reduced(input_shape.rank(), false);
            for (const Shape::size_type axis : axes)
            {
                is_reduced[axis] = true;
            }

            std::vector<Dimension> outer_dimensions;
            std::vector<Dimension> reduction_dimensions;
            outer_dimensions.reserve(input_shape.rank() - axes.size());
            reduction_dimensions.reserve(axes.size());

            for (Shape::size_type axis = 0; axis < input_shape.rank(); ++axis)
            {
                const Shape::size_type extent =
                    static_cast<Shape::size_type>(input_shape[axis]);
                if (extent == 1)
                {
                    continue;
                }

                const Layout::stride_type step = input_layout.stride(axis);
                const Dimension dimension{
                    extent,
                    step,
                    step * static_cast<Layout::offset_type>(extent - 1)};
                if (is_reduced[axis])
                {
                    reduction_dimensions.push_back(dimension);
                }
                else
                {
                    outer_dimensions.push_back(dimension);
                }
            }

            const T *input_data = data<T>(input);
            T *output_data = data<T>(output);
            const auto output_offset = static_cast<std::size_t>(output.layout().offset());
            Layout::offset_type outer_offset = input_layout.offset();
            std::vector<Shape::size_type> outer_indices(
                outer_dimensions.size(), Shape::size_type{0});
            std::vector<Shape::size_type> reduction_indices(
                reduction_dimensions.size(), Shape::size_type{0});

            for (Shape::size_type output_linear = 0;
                 output_linear < output.shape().numel();
                 ++output_linear)
            {
                T accumulator{};

                if (reduction_dimensions.empty())
                {
                    assert(outer_offset >= 0);
                    accumulator = input_data[static_cast<std::size_t>(outer_offset)];
                }
                else if (reduction_dimensions.size() == 1)
                {
                    Layout::offset_type input_offset = outer_offset;
                    const Layout::stride_type step = reduction_dimensions.front().step;
                    for (Shape::size_type i = 0; i < values_per_output; ++i)
                    {
                        assert(input_offset >= 0);
                        accumulator += input_data[static_cast<std::size_t>(input_offset)];
                        input_offset += step;
                    }
                }
                else
                {
                    std::fill(
                        reduction_indices.begin(),
                        reduction_indices.end(),
                        Shape::size_type{0});
                    Layout::offset_type input_offset = outer_offset;
                    for (Shape::size_type i = 0; i < values_per_output; ++i)
                    {
                        assert(input_offset >= 0);
                        accumulator += input_data[static_cast<std::size_t>(input_offset)];
                        if (i + 1 < values_per_output)
                        {
                            advance(reduction_indices, reduction_dimensions, input_offset);
                        }
                    }
                }

                output_data[output_offset + output_linear] = accumulator;
                if (output_linear + 1 < output.shape().numel())
                {
                    advance(outer_indices, outer_dimensions, outer_offset);
                }
            }
        }

        template <CpuElement T>
        void sum_values(
            const TensorView &input,
            MutableTensorView output,
            std::span<const Shape::size_type> axes)
        {
            assert(input.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
            assert(output.layout().is_contiguous(output.shape()));

            if (output.shape().numel() == 0)
            {
                return;
            }

            T *output_data = data<T>(output);
            const auto output_offset = static_cast<std::size_t>(output.layout().offset());
            const Shape::size_type values_per_output = reduction_size(input.shape(), axes);

            if (values_per_output == 0)
            {
                std::fill_n(output_data + output_offset, output.shape().numel(), T{});
                return;
            }

            if (input.layout().is_contiguous(input.shape()) &&
                reduces_contiguous_suffix(input.shape().rank(), axes))
            {
                sum_contiguous_suffix<T>(input, output, values_per_output);
                return;
            }

            sum_strided<T>(input, output, axes, values_per_output);
        }

        void run_sum(
            DeviceRuntime &,
            const Primitive &primitive,
            std::span<const TensorView> inputs,
            MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &sum = dynamic_cast<const SumPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    sum_values<T>(inputs.front(), output, sum.axes());
                });
        }
    }

    void register_reduction_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(
            KernelKey{typeid(SumPrimitive), DeviceType::Cpu},
            run_sum);
    }
}
