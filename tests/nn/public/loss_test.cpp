#include <minitensor/nn/loss.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
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

	TEST(L1LossTest, ComputesLossAndPredictionSubgradient)
	{
		const Tensor prediction = from_data(
			{ 1.0F, 2.0F, 0.0F, -2.0F }, Shape{ 4 });
		const Tensor target = from_data(
			{ 0.0F, 2.0F, 1.0F, 1.0F }, Shape{ 4 });
		const Tensor loss = l1_loss(prediction, target);

		EXPECT_TRUE(loss.shape().is_scalar());
		EXPECT_NEAR(item(loss), 5.0F / 4.0F, 1.0E-6F);

		const std::array<float, 4> expected_gradient{
			0.25F, 0.0F, -0.25F, -0.25F
		};
		minitensor::test::expect_tensor_near(
			grad(loss, prediction).front(), expected_gradient);
	}

	TEST(BinaryCrossEntropyWithLogitsTest, ComputesStableLossAndLogitsGradient)
	{
		const Tensor logits = from_data(
			{ -100.0F, -2.0F, 0.0F, 2.0F, 100.0F }, Shape{ 5 });
		const Tensor target = from_data(
			{ 0.0F, 0.0F, 1.0F, 1.0F, 1.0F }, Shape{ 5 });
		const Tensor loss = binary_cross_entropy_with_logits(logits, target);

		EXPECT_TRUE(loss.shape().is_scalar());
		EXPECT_NEAR(item(loss), 0.1894006F, 1.0E-6F);

		const std::array<float, 5> expected_gradient{
			0.0F, 0.0238406F, -0.1F, -0.0238406F, 0.0F
		};
		minitensor::test::expect_tensor_near(
			grad(loss, logits).front(), expected_gradient);
	}

	TEST(BinaryCrossEntropyWithLogitsTest, RejectsDifferentShapes)
	{
		const Tensor logits = full(Shape{ 2 }, 0.0F);
		const Tensor target = full(Shape{ 2, 1 }, 0.0F);

		EXPECT_THROW(
			(void)binary_cross_entropy_with_logits(logits, target),
			std::invalid_argument);
	}

	TEST(CategoricalCrossEntropyWithLogitsTest, ComputesLossAndLogitsGradient)
	{
		const Tensor logits = from_data(
			{ 2.0F, 0.0F, 1.0F, 1.0F, 0.0F, 2.0F }, Shape{ 3, 2 });
		const Tensor targets =
			from_data<std::int64_t>({ 0, 1 }, Shape{ 2 });
		const Tensor loss =
			categorical_cross_entropy_with_logits(logits, targets, 0);

		EXPECT_TRUE(loss.shape().is_scalar());
		EXPECT_NEAR(item(loss), 0.9076060F, 1.0E-6F);

		const std::array<float, 6> expected_gradient{
			-0.1673795F, 0.0450153F,
			0.1223642F, -0.3776358F,
			0.0450153F, 0.3326205F
		};
		minitensor::test::expect_tensor_near(
			grad(loss, logits).front(), expected_gradient);
	}
}
