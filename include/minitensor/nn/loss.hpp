#pragma once

#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	[[nodiscard]] Tensor mse_loss(const Tensor& prediction, const Tensor& target);
}