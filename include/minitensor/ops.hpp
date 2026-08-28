#pragma once

#include "tensor.hpp"
#include "types.hpp"

namespace minitensor
{
    // Minimal initial set of operations

    [[nodiscard]] Tensor operator+(const Tensor &lhs, const Tensor &rhs);
    [[nodiscard]] Tensor full(Shape shape, float value, TensorOptions options = {});
} // namespace minitensor
