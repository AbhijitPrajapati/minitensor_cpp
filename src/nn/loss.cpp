#include <minitensor/nn/loss.hpp>

#include <stdexcept>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	Tensor mse_loss(const Tensor& prediction, const Tensor& target)
	{
		if (prediction.shape() != target.shape())
		{
			throw std::invalid_argument{
				"mse loss requires prediction and target to have the same shape" };
		}
		Tensor error = prediction - target;
		return mean(error * error);
	}

	Tensor l1_loss(const Tensor& prediction, const Tensor& target)
	{
		if (prediction.shape() != target.shape())
		{
			throw std::invalid_argument{
				"l1 loss requires prediction and target to have the same shape" };
		}
		return mean(abs(prediction - target));
	}

	Tensor binary_cross_entropy_with_logits(const Tensor& logits, const Tensor& target)
	{
		if (logits.shape() != target.shape())
		{
			throw std::invalid_argument{
				"binary cross entropy requires logits and target to have the same shape" };
		}
		return mean(logaddexp(0.0F, logits) - target * logits);
	}
}
