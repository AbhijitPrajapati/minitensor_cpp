#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	namespace
	{
		enum class BinaryOperation
		{
			Add,
			Subtract,
			Multiply,
			Divide
		};

		Tensor apply(BinaryOperation operation, const Tensor& lhs, const Tensor& rhs)
		{
			switch (operation)
			{
			case BinaryOperation::Add:
				return lhs + rhs;
			case BinaryOperation::Subtract:
				return lhs - rhs;
			case BinaryOperation::Multiply:
				return lhs * rhs;
			case BinaryOperation::Divide:
				return lhs / rhs;
			}
			throw std::logic_error{ "unknown test operation" };
		}

		std::string binary_name(const testing::TestParamInfo<BinaryOperation>& info)
		{
			switch (info.param)
			{
			case BinaryOperation::Add:
				return "Add";
			case BinaryOperation::Subtract:
				return "Subtract";
			case BinaryOperation::Multiply:
				return "Multiply";
			case BinaryOperation::Divide:
				return "Divide";
			}
			return "Unknown";
		}

		enum class UnaryOperation
		{
			Negate,
			Exponential,
			Logarithm,
			SquareRoot,
			HyperbolicTangent
		};

		Tensor apply(UnaryOperation operation, const Tensor& input)
		{
			switch (operation)
			{
			case UnaryOperation::Negate:
				return -input;
			case UnaryOperation::Exponential:
				return exp(input);
			case UnaryOperation::Logarithm:
				return log(input);
			case UnaryOperation::SquareRoot:
				return sqrt(input);
			case UnaryOperation::HyperbolicTangent:
				return tanh(input);
			}
			throw std::logic_error{ "unknown test operation" };
		}

		std::string unary_name(const testing::TestParamInfo<UnaryOperation>& info)
		{
			switch (info.param)
			{
			case UnaryOperation::Negate:
				return "Negate";
			case UnaryOperation::Exponential:
				return "Exponential";
			case UnaryOperation::Logarithm:
				return "Logarithm";
			case UnaryOperation::SquareRoot:
				return "SquareRoot";
			case UnaryOperation::HyperbolicTangent:
				return "HyperbolicTangent";
			}
			return "Unknown";
		}
	}

	class BinaryElementwiseTest : public testing::TestWithParam<BinaryOperation>
	{};

	TEST_P(BinaryElementwiseTest, BroadcastsAndComputesValues)
	{
		const std::array<float, 2> lhs_values{ 2.0F, 4.0F };
		const std::array<float, 3> rhs_values{ 1.0F, 2.0F, 4.0F };
		const Tensor lhs = from_data(lhs_values, Shape{ 2, 1 });
		const Tensor rhs = from_data(rhs_values, Shape{ 1, 3 });
		const Tensor result = apply(GetParam(), lhs, rhs);

		EXPECT_EQ(result.shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(result.dtype(), DType::Float32);
		EXPECT_EQ(result.device(), Device::cpu());

		switch (GetParam())
		{
		case BinaryOperation::Add:
		{
			const std::array<float, 6> expected{ 3.0F, 4.0F, 6.0F, 5.0F, 6.0F, 8.0F };
			expect_tensor_eq(result, expected);
			break;
		}
		case BinaryOperation::Subtract:
		{
			const std::array<float, 6> expected{ 1.0F, 0.0F, -2.0F, 3.0F, 2.0F, 0.0F };
			expect_tensor_eq(result, expected);
			break;
		}
		case BinaryOperation::Multiply:
		{
			const std::array<float, 6> expected{ 2.0F, 4.0F, 8.0F, 4.0F, 8.0F, 16.0F };
			expect_tensor_eq(result, expected);
			break;
		}
		case BinaryOperation::Divide:
		{
			const std::array<float, 6> expected{ 2.0F, 1.0F, 0.5F, 4.0F, 2.0F, 1.0F };
			expect_tensor_near(result, expected);
			break;
		}
		}
	}

	TEST_P(BinaryElementwiseTest, RejectsIncompatibleShapes)
	{
		const Tensor lhs = full(Shape{ 2, 3 }, 1.0F);
		const Tensor rhs = full(Shape{ 2, 2 }, 1.0F);
		EXPECT_THROW((void)apply(GetParam(), lhs, rhs), std::invalid_argument);
	}

	TEST_P(BinaryElementwiseTest, RejectsDifferentDevices)
	{
		const Tensor lhs = full(
			Shape{ 2 }, 1.0F,
			TensorOptions{ DType::Float32, Device::cpu(0) });
		const Tensor rhs = full(
			Shape{ 2 }, 1.0F,
			TensorOptions{ DType::Float32, Device::cpu(1) });
		EXPECT_THROW((void)apply(GetParam(), lhs, rhs), std::invalid_argument);
	}

	INSTANTIATE_TEST_SUITE_P(
		AllOperations,
		BinaryElementwiseTest,
		testing::Values(
			BinaryOperation::Add,
			BinaryOperation::Subtract,
			BinaryOperation::Multiply,
			BinaryOperation::Divide),
		binary_name);

	class UnaryElementwiseTest : public testing::TestWithParam<UnaryOperation>
	{};

	TEST_P(UnaryElementwiseTest, PreservesSpecificationAndHandlesStridedInputs)
	{
		const std::array<float, 4> values{ 1.0F, 2.0F, 4.0F, 8.0F };
		const Tensor input = transpose(from_data(values, Shape{ 2, 2 }));
		const Tensor result = apply(GetParam(), input);

		EXPECT_EQ(result.shape(), input.shape());
		EXPECT_EQ(result.dtype(), input.dtype());
		EXPECT_EQ(result.device(), input.device());

		const std::array<float, 4> logical_input{ 1.0F, 4.0F, 2.0F, 8.0F };
		std::array<float, 4> expected{};
		for (std::size_t index = 0; index < expected.size(); ++index)
		{
			switch (GetParam())
			{
			case UnaryOperation::Negate:
				expected[index] = -logical_input[index];
				break;
			case UnaryOperation::Exponential:
				expected[index] = std::exp(logical_input[index]);
				break;
			case UnaryOperation::Logarithm:
				expected[index] = std::log(logical_input[index]);
				break;
			case UnaryOperation::SquareRoot:
				expected[index] = std::sqrt(logical_input[index]);
				break;
			case UnaryOperation::HyperbolicTangent:
				expected[index] = std::tanh(logical_input[index]);
				break;
			}
		}
		expect_tensor_near(result, expected);
	}

	INSTANTIATE_TEST_SUITE_P(
		AllOperations,
		UnaryElementwiseTest,
		testing::Values(
			UnaryOperation::Negate,
			UnaryOperation::Exponential,
			UnaryOperation::Logarithm,
			UnaryOperation::SquareRoot,
			UnaryOperation::HyperbolicTangent),
		unary_name);

	TEST(ElementwiseScalarOverloadTest, PreservesOperandOrder)
	{
		const std::array<float, 3> values{ 1.0F, 2.0F, 4.0F };
		const Tensor input = from_data(values, Shape{ 3 });

		const std::array<float, 3> add{ 3.0F, 4.0F, 6.0F };
		const std::array<float, 3> tensor_subtract{ -1.0F, 0.0F, 2.0F };
		const std::array<float, 3> scalar_subtract{ 1.0F, 0.0F, -2.0F };
		const std::array<float, 3> multiply{ 2.0F, 4.0F, 8.0F };
		const std::array<float, 3> tensor_divide{ 0.5F, 1.0F, 2.0F };
		const std::array<float, 3> scalar_divide{ 8.0F, 4.0F, 2.0F };

		expect_tensor_eq(input + 2.0F, add);
		expect_tensor_eq(2.0F + input, add);
		expect_tensor_eq(input - 2.0F, tensor_subtract);
		expect_tensor_eq(2.0F - input, scalar_subtract);
		expect_tensor_eq(input * 2.0F, multiply);
		expect_tensor_eq(2.0F * input, multiply);
		expect_tensor_near(input / 2.0F, tensor_divide);
		expect_tensor_near(8.0F / input, scalar_divide);
	}

	TEST(ElementwiseTest, SupportsScalarAndEmptyBroadcasting)
	{
		EXPECT_EQ(
			(full(Shape{ 2, 3 }, 1.0F) + full(Shape{}, 2.0F)).shape(),
			(Shape{ 2, 3 }));
		EXPECT_EQ(
			(full(Shape{ 2, 0, 3 }, 1.0F) + full(Shape{ 1, 3 }, 2.0F)).shape(),
			(Shape{ 2, 0, 3 }));
	}
}
