#pragma once

#include <minitensor/tensor.hpp>

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
}
