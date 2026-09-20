#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/iteration/elementwise.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/backend/cpu/kernels/common/buffer_access.hpp"

namespace minitensor::detail::cpu
{
    template <CpuElement T, typename Operation>
        requires std::invocable<Operation &, T> && std::convertible_to<std::invoke_result_t<Operation &, T>, T>
    void unary_elementwise(const TensorView &input, MutableTensorView output, Operation &&operation)
    {
        assert(input.dtype() == ElementDType<T>::value);
        assert(output.dtype() == ElementDType<T>::value);

        const Shape &output_shape = output.shape();
        assert(input.shape() == output_shape);
        assert(output.layout().is_contiguous(output_shape));

        if (output_shape.numel() == 0)
        {
            return;
        }

        const T *input_data = data<T>(input);
        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        const std::array<Layout, 1> layouts{input.layout()};
        const ElementwisePlan plan(output_shape, layouts);
        plan.for_each_run(
            [&](Shape::size_type linear, std::span<const Layout::offset_type> offsets, std::span<const Layout::stride_type> strides, Shape::size_type run_size)
            {
                assert(offsets.size() == 1);
                assert(strides.size() == 1);

                Layout::offset_type input_offset = offsets[0];
                const Layout::stride_type input_stride = strides[0];

                for (Shape::size_type i = 0; i < run_size; ++i)
                {
                    assert(input_offset >= 0);
                    output_data[output_offset + linear + i] = static_cast<T>(std::invoke(
                        operation,
                        input_data[static_cast<std::size_t>(input_offset)]));
                    input_offset += input_stride;
                }
            });
    }

    template <CpuElement T, typename Operation>
        requires std::invocable<Operation &, T, T> && std::convertible_to<std::invoke_result_t<Operation &, T, T>, T>
    void binary_elementwise(const TensorView &lhs, const TensorView &rhs, MutableTensorView output, Operation &&operation)
    {
        assert(lhs.dtype() == ElementDType<T>::value);
        assert(rhs.dtype() == ElementDType<T>::value);
        assert(output.dtype() == ElementDType<T>::value);

        const Shape &output_shape = output.shape();
        assert(output.layout().is_contiguous(output_shape));

        if (output_shape.numel() == 0)
        {
            return;
        }

        const T *lhs_data = data<T>(lhs);
        const T *rhs_data = data<T>(rhs);
        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        const std::array<Layout, 2> layouts{
            lhs.layout().broadcasted_to(lhs.shape(), output_shape),
            rhs.layout().broadcasted_to(rhs.shape(), output_shape)};

        const ElementwisePlan plan(output_shape, layouts);
        plan.for_each_run(
            [&](Shape::size_type linear, std::span<const Layout::offset_type> offsets, std::span<const Layout::stride_type> strides, Shape::size_type run_size)
            {
                assert(offsets.size() == 2);
                assert(strides.size() == 2);

                Layout::offset_type lhs_offset = offsets[0];
                Layout::offset_type rhs_offset = offsets[1];
                const Layout::stride_type lhs_stride = strides[0];
                const Layout::stride_type rhs_stride = strides[1];

                for (Shape::size_type i = 0; i < run_size; ++i)
                {
                    assert(lhs_offset >= 0);
                    assert(rhs_offset >= 0);
                    output_data[output_offset + linear + i] = static_cast<T>(std::invoke(
                        operation,
                        lhs_data[static_cast<std::size_t>(lhs_offset)],
                        rhs_data[static_cast<std::size_t>(rhs_offset)]));
                    lhs_offset += lhs_stride;
                    rhs_offset += rhs_stride;
                }
            });
    }
}
