#pragma once

#include "minitensor/tensor.hpp"

namespace minitensor {

// Operators are defined as free functions rather than Tensor methods

[[nodiscard]] Tensor operator+(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator-(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator*(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator/(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor operator-(const Tensor& value);

[[nodiscard]] Tensor operator+(const Tensor& tensor, float scalar);
[[nodiscard]] Tensor operator+(float scalar, const Tensor& tensor);
[[nodiscard]] Tensor operator-(const Tensor& tensor, float scalar);
[[nodiscard]] Tensor operator-(float scalar, const Tensor& tensor);
[[nodiscard]] Tensor operator*(const Tensor& tensor, float scalar);
[[nodiscard]] Tensor operator*(float scalar, const Tensor& tensor);
[[nodiscard]] Tensor operator/(const Tensor& tensor, float scalar);
[[nodiscard]] Tensor operator/(float scalar, const Tensor& tensor);

[[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs);
[[nodiscard]] Tensor relu(const Tensor& input);
[[nodiscard]] Tensor sigmoid(const Tensor& input);
[[nodiscard]] Tensor tanh(const Tensor& input);

} // namespace minitensor
