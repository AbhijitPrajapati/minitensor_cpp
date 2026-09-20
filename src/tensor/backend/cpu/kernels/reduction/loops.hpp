#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/buffer_access.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/iteration/reduction.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    template <CpuElement T, typename Combine>
        requires std::invocable<Combine &, T, T> && std::convertible_to<std::invoke_result_t<Combine &, T, T>, T>
    void reduce(const TensorView &input, MutableTensorView output, std::span<const Shape::size_type> axes, T identity, Combine &&combine)
    {
        assert(input.dtype() == ElementDType<T>::value);
        assert(output.dtype() == ElementDType<T>::value);
        assert(output.layout().is_contiguous(output.shape()));

        if (output.shape().numel() == 0)
        {
            return;
        }

        const T *input_data = data<T>(input);
        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        const ReductionPlan plan(input.shape(), input.layout(), output.shape(), axes);
        std::fill_n(output_data + output_offset, plan.output_numel(), identity);

        plan.for_each(
            [&](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                assert(input_offset >= 0);
                T &accumulator = output_data[output_offset + output_linear];
                accumulator = static_cast<T>(std::invoke(
                    combine,
                    accumulator,
                    input_data[static_cast<std::size_t>(input_offset)]));
            });
    }

    template <CpuElement T, typename Combine, typename Finalize>
        requires std::invocable<Combine &, T, T> &&
                 std::convertible_to<std::invoke_result_t<Combine &, T, T>, T> &&
                 std::invocable<Finalize &, T> &&
                 std::convertible_to<std::invoke_result_t<Finalize &, T>, T>
    void reduce(
        const TensorView &input,
        MutableTensorView output,
        std::span<const Shape::size_type> axes,
        T identity,
        Combine &&combine,
        Finalize &&finalize)
    {
        reduce<T>(input, output, axes, identity, std::forward<Combine>(combine));

        const Shape::size_type output_numel = output.shape().numel();
        if (output_numel == 0)
        {
            return;
        }

        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());
        for (Shape::size_type index = 0; index < output_numel; ++index)
        {
            T &value = output_data[output_offset + index];
            value = static_cast<T>(std::invoke(finalize, value));
        }
    }
}
