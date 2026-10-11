#include <minitensor/nn/loss.hpp>

#include <minitensor/tensor.hpp>
#include <minitensor/ops.hpp>

namespace minitensor::nn
{
	Tensor mse_loss(const Tensor& prediction, const Tensor& target)
	{
		Tensor error = prediction - target;
		return mean(error * error);
	}
}