#include <minitensor/data.hpp>
#include <minitensor/evaluation.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <span>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(EvaluationTest, AcceptsNoRoots)
	{
		EXPECT_NO_THROW(eval(std::span<const Tensor>{}));
	}

	TEST(EvaluationTest, EvaluatesSingleAndSharedRoots)
	{
		const Tensor lhs = full(Shape{ 2, 1 }, 1.5F);
		const Tensor rhs = full(Shape{ 1, 3 }, 2.5F);
		const Tensor sum = lhs + rhs;
		const Tensor result = sum + 4.0F;
		const std::array<Tensor, 4> roots{ lhs, sum, result, result };

		EXPECT_NO_THROW(eval(roots));
		const std::array<float, 6> expected{ 8.0F, 8.0F, 8.0F, 8.0F, 8.0F, 8.0F };
		expect_tensor_eq(result, expected);
		EXPECT_NO_THROW(eval(result));
	}

	TEST(EvaluationTest, SupportsInitializerListsAndEmptyStorage)
	{
		const Tensor scalar = full(Shape{}, 3.5F);
		const Tensor empty = full(Shape{ 2, 0, 3 }, 7.0F);

		EXPECT_NO_THROW(eval({ scalar, empty }));
		EXPECT_FLOAT_EQ(item(scalar), 3.5F);
		EXPECT_TRUE(to_vector(empty).empty());
	}

	TEST(EvaluationTest, ReportsUnavailableDevices)
	{
		const Tensor unsupported = full(
			Shape{ 1 }, 1.0F,
			TensorOptions{ DType::Float32, Device::cpu(1) });
		EXPECT_THROW(eval(unsupported), std::runtime_error);
	}
}
