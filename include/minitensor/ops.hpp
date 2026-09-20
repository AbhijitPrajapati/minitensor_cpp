#pragma once

#include <span>

#include "tensor.hpp"
#include "types.hpp"

namespace minitensor
{
    [[nodiscard]] Tensor operator+(const Tensor &lhs, const Tensor &rhs);
    [[nodiscard]] Tensor operator-(const Tensor &input);
    [[nodiscard]] Tensor operator-(const Tensor &lhs, const Tensor &rhs);
    [[nodiscard]] Tensor operator*(const Tensor &lhs, const Tensor &rhs);
    [[nodiscard]] Tensor operator/(const Tensor &lhs, const Tensor &rhs);
    [[nodiscard]] Tensor exp(const Tensor &input);
    [[nodiscard]] Tensor log(const Tensor &input);
    [[nodiscard]] Tensor sqrt(const Tensor &input);
    [[nodiscard]] Tensor tanh(const Tensor &input);
    [[nodiscard]] Tensor full(Shape shape, float value, TensorOptions options = {});
    [[nodiscard]] Tensor full_like(const Tensor &input, float value);
    [[nodiscard]] Tensor full_like(const Tensor &input, float value, TensorOptions options);
    [[nodiscard]] Tensor zeros(Shape shape, TensorOptions options = {});
    [[nodiscard]] Tensor ones(Shape shape, TensorOptions options = {});
    [[nodiscard]] Tensor zeros_like(const Tensor &input);
    [[nodiscard]] Tensor zeros_like(const Tensor &input, TensorOptions options);
    [[nodiscard]] Tensor ones_like(const Tensor &input);
    [[nodiscard]] Tensor ones_like(const Tensor &input, TensorOptions options);
    [[nodiscard]] Tensor permute(const Tensor &input, std::span<const Axis> permutation);
    [[nodiscard]] Tensor transpose(const Tensor &input);
    [[nodiscard]] Tensor transpose(const Tensor &input, Axis axis0, Axis axis1);
    [[nodiscard]] Tensor reshape(const Tensor &input, Shape shape);
    [[nodiscard]] Tensor flatten(const Tensor &input);
    [[nodiscard]] Tensor flatten(const Tensor &input, Axis start_axis, Axis end_axis = -1);
    [[nodiscard]] Tensor squeeze(const Tensor &input);
    [[nodiscard]] Tensor squeeze(const Tensor &input, Axis axis);
    [[nodiscard]] Tensor squeeze(const Tensor &input, std::span<const Axis> axes);
    [[nodiscard]] Tensor unsqueeze(const Tensor &input, Axis axis);
    [[nodiscard]] Tensor broadcast_to(const Tensor &input, Shape shape);
    [[nodiscard]] Tensor sum(const Tensor &input, std::span<const Axis> axes, bool keep_dim = false);
    [[nodiscard]] Tensor sum(const Tensor &input, bool keep_dim = false);
    [[nodiscard]] Tensor mean(const Tensor &input, std::span<const Axis> axes, bool keep_dim = false);
    [[nodiscard]] Tensor mean(const Tensor &input, bool keep_dim = false);
    [[nodiscard]] Tensor max(const Tensor &input, std::span<const Axis> axes, bool keep_dim = false);
    [[nodiscard]] Tensor max(const Tensor &input, bool keep_dim = false);
    [[nodiscard]] Tensor min(const Tensor &input, std::span<const Axis> axes, bool keep_dim = false);
    [[nodiscard]] Tensor min(const Tensor &input, bool keep_dim = false);
    [[nodiscard]] Tensor matmul(const Tensor &lhs, const Tensor &rhs);
} // namespace minitensor
