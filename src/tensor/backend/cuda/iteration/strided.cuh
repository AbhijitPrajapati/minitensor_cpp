#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace minitensor::detail::cuda
{
// Decode a row-major logical index once for all operands. Strides are packed operand-major: strides[operand * rank +
// axis]. Offsets initially contain each operand's storage offset and receive its final element offset. Call only for
// valid logical indices of a nonempty iteration shape; extents are therefore positive. Rank zero leaves the initial
// scalar offsets unchanged.
template <std::size_t OperandCount>
__device__ inline void strided_offsets(std::size_t linear, const std::int64_t *extents, const std::int64_t *strides,
                                       std::size_t rank, std::int64_t (&offsets)[OperandCount])
{
    static_assert(OperandCount > 0);
    for (std::size_t axis = rank; axis > 0;)
    {
        --axis;
        const auto extent = static_cast<std::size_t>(extents[axis]);
        const auto coordinate = static_cast<std::int64_t>(linear % extent);
        linear /= extent;
        for (std::size_t operand = 0; operand < OperandCount; ++operand)
        {
            offsets[operand] += coordinate * strides[operand * rank + axis];
        }
    }
}
} // namespace minitensor::detail::cuda
