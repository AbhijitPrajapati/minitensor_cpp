#pragma once

#include <cassert>
#include <array>
#include <cstddef>
#include <concepts>
#include <functional>
#include <span>
#include <algorithm>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/iteration/reduction.hpp"
#include "../cpu_buffer.hpp"

namespace minitensor::detail::cpu
{
    template <typename Combine>
        requires std::invocable<Combine &, float, float>
    void reduction_float32(
        const TensorView &input,
        MutableTensorView output,
        std::span<const Shape::size_type> axes,
        float identity,
        Combine &&combine)
    {
        assert(input.dtype() == DType::Float32);
        assert(output.dtype() == DType::Float32);

        const Shape output_shape = output.shape();
        assert(output.layout().is_contiguous(output_shape));

        if (output_shape.numel() == 0)
        {
            return;
        }

        const auto &input_buffer = dynamic_cast<const CpuBuffer &>(input.buffer());
        auto &output_buffer = dynamic_cast<CpuBuffer &>(output.buffer());

        const auto *input_data = reinterpret_cast<const float *>(input_buffer.data());
        auto *output_data = reinterpret_cast<float *>(output_buffer.data());

        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        const ReductionPlan plan(input.shape(), input.layout(), output_shape, axes);

        // initialize output with identity
        std::fill_n(output_data + output_offset, plan.output_numel(), identity);

        plan.for_each(
            [&](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                float &accumulator = output_data[output_linear + output_offset];
                const auto input_idx = static_cast<std::size_t>(input_offset);
                const float value = input_data[input_idx];
                accumulator = std::invoke(combine, accumulator, value);
            });
    }
}
