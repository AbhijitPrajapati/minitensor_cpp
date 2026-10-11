#include <minitensor/nn/loss.hpp>

#include <array>

#include <gtest/gtest.h>

#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::nn::test
{
	TEST(MseLossTest, ComputesLossAndPredictionGradient)
	{
		const Tensor prediction = from_data(
			{ 1.0F, 2.0F, 4.0F }, Shape{ 3 });
		const Tensor target = from_data(
			{ 0.0F, 2.0F, 1.0F }, Shape{ 3 });
		const Tensor loss = mse_loss(prediction, target);

		EXPECT_TRUE(loss.shape().is_scalar());
		EXPECT_NEAR(item(loss), 10.0F / 3.0F, 1.0E-6F);

		const std::array<float, 3> expected_gradient{
			2.0F / 3.0F, 0.0F, 2.0F
		};
		minitensor::test::expect_tensor_near(
			grad(loss, prediction).front(), expected_gradient);
	}
}
