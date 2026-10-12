#include <minitensor/nn/loss.hpp>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	Tensor mse_loss(const Tensor& prediction, const Tensor& target)
	{
		Tensor error = prediction - target;
		return mean(error * error);
	}

	Tensor l1_loss(const Tensor& prediction, const Tensor& target)
	{
		return mean(abs(prediction - target));
	}
}