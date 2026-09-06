#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "../cpu_buffer.hpp"

namespace minitensor::detail::cpu
{
    template <typename Function>
    void for_each_binary_offset(const Shape &shape, const Layout &lhs, const Layout &rhs, Function &&function)
    {
        struct AxisState
        {
            Extent extent;
            Extent index;
            std::int64_t lhs_step;
            std::int64_t rhs_step;
            std::int64_t lhs_reset;
            std::int64_t rhs_reset;
        };

        std::vector<AxisState> axes;
        axes.reserve(shape.rank());

        for (Shape::size_type axis = 0; axis < shape.rank(); ++axis)
        {
            const Extent extent = shape[axis];

            if (extent == 1)
            {
                continue;
            }

            const auto lhs_step = lhs.stride(axis);
            const auto rhs_step = rhs.stride(axis);

            axes.push_back(AxisState{
                .extent = extent,
                .index = 0,
                .lhs_step = lhs_step,
                .rhs_step = rhs_step,
                .lhs_reset = lhs_step * (extent - 1),
                .rhs_reset = rhs_step * (extent - 1),
            });
        }

        auto lhs_offset = lhs.offset();
        auto rhs_offset = rhs.offset();

        const std::size_t count = shape.numel();

        for (Shape::size_type linear = 0; linear < count; ++linear)
        {
            std::invoke(
                function,
                linear,
                lhs_offset,
                rhs_offset);

            if (linear + 1 == count)
            {
                break;
            }

            for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis)
            {
                if (axis->index + 1 < axis->extent)
                {
                    ++axis->index;
                    lhs_offset += axis->lhs_step;
                    rhs_offset += axis->rhs_step;
                    break;
                }

                axis->index = 0;
                lhs_offset -= axis->lhs_reset;
                rhs_offset -= axis->rhs_reset;
            }
        }
    }

    template <typename Operation>
    void binary_elementwise_float32(
        const TensorView &lhs,
        const TensorView &rhs,
        MutableTensorView output,
        Operation &&operation)
    {

        assert(lhs.dtype() == DType::Float32);
        assert(rhs.dtype() == DType::Float32);
        assert(output.dtype() == DType::Float32);

        const Shape output_shape = output.shape();
        assert(output.layout().is_contiguous(output_shape));

        const std::size_t numel = output_shape.numel();
        if (numel == 0)
        {
            return;
        }

        const auto &lhs_buffer = dynamic_cast<const CpuBuffer &>(lhs.buffer());
        const auto &rhs_buffer = dynamic_cast<const CpuBuffer &>(rhs.buffer());
        auto &output_buffer = dynamic_cast<CpuBuffer &>(output.buffer());

        const auto *lhs_data = reinterpret_cast<const float *>(lhs_buffer.data());
        const auto *rhs_data = reinterpret_cast<const float *>(rhs_buffer.data());
        auto *output_data = reinterpret_cast<float *>(output_buffer.data());

        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        const Layout lhs_layout = lhs.layout().broadcasted_to(lhs.shape(), output.shape());
        const Layout rhs_layout = rhs.layout().broadcasted_to(rhs.shape(), output.shape());

        for_each_binary_offset(
            output_shape,
            lhs_layout,
            rhs_layout,
            [&](std::size_t linear,
                Layout::offset_type lhs_offset,
                Layout::offset_type rhs_offset)
            {
                assert(lhs_offset >= 0);
                assert(rhs_offset >= 0);

                const auto lhs_index = static_cast<std::size_t>(lhs_offset);
                const auto rhs_index = static_cast<std::size_t>(rhs_offset);

                output_data[linear + output_offset] = std::invoke(
                    operation,
                    lhs_data[lhs_index],
                    rhs_data[rhs_index]);
            });
    }
}