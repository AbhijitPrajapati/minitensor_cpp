#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

#include "tensor/backend/cpu/kernels/common/buffer_access.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/iteration/elementwise.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    template <CpuElement T>
    void copy_to_dense(const TensorView &input, MutableTensorView output)
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
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        const std::array<Layout, 1> layouts{input.layout()};
        const ElementwisePlan plan(input.shape(), layouts);
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
                    output_data[output_offset + linear + i] = input_data[static_cast<std::size_t>(input_offset)];
                    input_offset += input_stride;
                }
            });
    }
}
