#include <minitensor/nn/optimizers/sgd.hpp>

#include <cmath>
#include <stdexcept>

namespace minitensor::nn
{
	SGD::SGD(float learning_rate) : learning_rate_(learning_rate)
	{
		if (!std::isfinite(learning_rate_) || learning_rate_ < 0.0F)
		{
			throw std::invalid_argument{ "SGD learning rate must be finite and non-negative" };
		}
	}

	float SGD::learning_rate() const noexcept
	{
		return learning_rate_;
	}
}
