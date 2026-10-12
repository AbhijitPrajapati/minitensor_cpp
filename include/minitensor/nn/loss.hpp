#pragma once

#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	[[nodiscard]] Tensor mse_loss(const Tensor& prediction, const Tensor& target);
	[[nodiscard]] Tensor l1_loss(const Tensor& prediction, const Tensor& target);
	[[nodiscard]] Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target);
}