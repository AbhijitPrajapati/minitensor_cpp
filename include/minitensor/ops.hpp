#pragma once

#include <span>

#include "tensor.hpp"
#include "types.hpp"

namespace minitensor
{
    [[nodiscard]] Tensor operator+(const Tensor &lhs, const Tensor &rhs);
    [[nodiscard]] Tensor full(Shape shape, float value, TensorOptions options = {});
    [[nodiscard]] Tensor permute(const Tensor &input, std::span<const Axis> permutation);
    [[nodiscard]] Tensor reshape(const Tensor &input, Shape shape);
    [[nodiscard]] Tensor broadcast_to(const Tensor &input, Shape shape);
} // namespace minitensor
