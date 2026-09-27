#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(AutogradTest, ReturnsCotangentsInRequestedOrderIncludingDisconnectedInputs)
	{
		const Tensor x = from_data({ 2.0F, 3.0F }, Shape{ 2 });
		const Tensor disconnected = full(Shape{ 2 }, 5.0F);
		const std::array<Tensor, 3> inputs{ disconnected, x, x };
		const std::vector<Tensor> gradients = grad(sum(x * x), inputs);

		ASSERT_EQ(gradients.size(), inputs.size());
		const std::array<float, 2> zero{};
		const std::array<float, 2> expected{ 4.0F, 6.0F };
		expect_tensor_eq(gradients[0], zero);
		expect_tensor_eq(gradients[1], expected);
		expect_tensor_eq(gradients[2], expected);
	}

	TEST(AutogradTest, ReturnsTheSeedWhenTheOutputIsTheTarget)
	{
		const Tensor output = full(Shape{ 2 }, 4.0F);
		const Tensor seed = from_data({ 1.5F, -2.0F }, Shape{ 2 });
		const std::vector<Tensor> result = vjp(output, output, seed);

		ASSERT_EQ(result.size(), 1);
		const std::array<float, 2> expected{ 1.5F, -2.0F };
		expect_tensor_eq(result.front(), expected);
	}

	TEST(AutogradTest, SupportsSecondDerivativesOfCompositeGraphs)
	{
		const Tensor x = from_data({ 3.0F }, Shape{});
		const Tensor cubic = x * x * x;
		const Tensor first = grad(cubic, x).front();
		const Tensor second = grad(first, x).front();

		EXPECT_FLOAT_EQ(item(first), 27.0F);
		EXPECT_FLOAT_EQ(item(second), 18.0F);
	}

	TEST(AutogradTest, SupportsSecondDerivativesThroughUnaryRules)
	{
		const Tensor x = from_data({ 1.0F, 2.0F }, Shape{ 2 });
		const Tensor first = grad(sum(exp(x)), x).front();
		const Tensor second = grad(sum(first), x).front();
		const std::array<float, 2> expected{ std::exp(1.0F), std::exp(2.0F) };

		expect_tensor_near(first, expected);
		expect_tensor_near(second, expected);
	}

	TEST(AutogradTest, ValidatesOutputCotangents)
	{
		const Tensor input = full(Shape{ 2, 3 }, 1.0F);
		const Tensor output = transpose(input);

		EXPECT_THROW(
			(void)vjp(output, input, full(Shape{ 1 }, 1.0F)),
			std::invalid_argument);
		EXPECT_THROW(
			(void)vjp(
				output,
				input,
				full(
					output.shape(), 1.0F,
					TensorOptions{ DType::Float32, Device::cpu(1) })),
			std::invalid_argument);
	}

	TEST(AutogradTest, GradRequiresAScalarOutput)
	{
		const Tensor input = full(Shape{ 2 }, 1.0F);
		EXPECT_THROW((void)grad(input, input), std::invalid_argument);
	}
}
