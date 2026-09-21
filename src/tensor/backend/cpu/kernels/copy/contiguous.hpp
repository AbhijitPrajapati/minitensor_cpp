#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

#include "tensor/backend/cpu/kernels/detail/buffer_access.hpp"
#include "tensor/backend/cpu/kernels/iteration/elementwise.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    template <CpuElement T>
    void copy_to_contiguous(const TensorView &input, MutableTensorView output)
    {
        assert(input.dtype() == ElementDType<T>::value);
        assert(output.dtype() == ElementDType<T>::value);
        assert(input.shape().numel() == output.shape().numel());
        assert(output.layout().is_contiguous(output.shape()));

        if (output.shape().numel() == 0)
        {
            return;
        }

        const T *input_data = data<T>(input);
        T *output_data = data<T>(output);
        const auto input_offset = static_cast<std::size_t>(input.layout().offset());
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());
        const Shape::size_type numel = output.shape().numel();

        if (input.layout().is_contiguous(input.shape()))
        {
            std::copy_n(input_data + input_offset, numel, output_data + output_offset);
            return;
        }

        if (is_single_value_layout(input.shape(), input.layout()))
        {
            std::fill_n(output_data + output_offset, numel, input_data[input_offset]);
            return;
        }

        const std::array<Layout, 1> layouts{input.layout()};
        const StridedIteration<1> iteration(input.shape(), layouts);
        iteration.for_each_run(
            [&](Shape::size_type linear,
                const StridedIteration<1>::Offsets &offsets,
                const StridedIteration<1>::Strides &strides,
                Shape::size_type run_size)
            {
                Layout::offset_type input_offset = offsets[0];
                const Layout::stride_type input_stride = strides[0];
                T *run_output = output_data + output_offset + linear;

                if (input_stride == 1)
                {
                    assert(input_offset >= 0);
                    std::copy_n(
                        input_data + static_cast<std::size_t>(input_offset),
                        run_size,
                        run_output);
                    return;
                }

                if (input_stride == 0)
                {
                    assert(input_offset >= 0);
                    std::fill_n(
                        run_output,
                        run_size,
                        input_data[static_cast<std::size_t>(input_offset)]);
                    return;
                }

                for (Shape::size_type i = 0; i < run_size; ++i)
                {
                    assert(input_offset >= 0);
                    run_output[i] = input_data[static_cast<std::size_t>(input_offset)];
                    input_offset += input_stride;
                }
            });
    }
}
