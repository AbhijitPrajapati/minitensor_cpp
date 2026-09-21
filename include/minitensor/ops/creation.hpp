#pragma once

#include <minitensor/tensor.hpp>

namespace minitensor
{
    [[nodiscard]] Tensor full(Shape shape, float value, TensorOptions options = {});
    [[nodiscard]] Tensor full_like(const Tensor &input, float value);
    [[nodiscard]] Tensor full_like(const Tensor &input, float value, TensorOptions options);
    [[nodiscard]] Tensor zeros(Shape shape, TensorOptions options = {});
    [[nodiscard]] Tensor ones(Shape shape, TensorOptions options = {});
    [[nodiscard]] Tensor zeros_like(const Tensor &input);
    [[nodiscard]] Tensor zeros_like(const Tensor &input, TensorOptions options);
    [[nodiscard]] Tensor ones_like(const Tensor &input);
    [[nodiscard]] Tensor ones_like(const Tensor &input, TensorOptions options);
}
