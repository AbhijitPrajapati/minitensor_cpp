#pragma once

namespace minitensor
{
    class Shape;
}

namespace minitensor::detail
{
    [[nodiscard]] Shape broadcast_shape(const Shape &lhs, const Shape &rhs);
    [[nodiscard]] Shape matmul_output_shape(const Shape &lhs, const Shape &rhs);
}
