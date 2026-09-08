#pragma once

#include <cassert>
#include <array>
#include <cstddef>
#include <concepts>
#include <functional>
#include <span>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/iteration/elementwise.hpp"
#include "../cpu_buffer.hpp"

namespace minitensor::detail::cpu
{
    template <typename Operation> requires std::invocable<Operation &, float, float>
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

        const std::array<Layout, 2> layouts{
            lhs.layout().broadcasted_to(lhs.shape(), output.shape()),
            rhs.layout().broadcasted_to(rhs.shape(), output.shape())
        };

        const ElementwisePlan plan(output_shape, layouts);
        plan.for_each(
            [&](Shape::size_type linear, std::span<const Layout::offset_type> offsets)
            {
                assert(offsets.size() == 2);
                assert(offsets[0] >= 0);
                assert(offsets[1] >= 0);
                const auto lhs_index = static_cast<std::size_t>(offsets[0]);
                const auto rhs_index = static_cast<std::size_t>(offsets[1]);
                output_data[linear + output_offset] = std::invoke(
                    operation,
                    lhs_data[lhs_index],
                    rhs_data[rhs_index]);}
        );
    }
}
