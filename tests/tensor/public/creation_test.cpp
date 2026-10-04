#include <minitensor/ops/creation.hpp>
#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(CreationTest, FullPreservesRequestedSpecification)
	{
		const Tensor matrix = full(Shape{ 2, 3 }, -2.5F);
		EXPECT_EQ(matrix.shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(matrix.dtype(), DType::Float32);
		EXPECT_EQ(matrix.device(), Device::cpu());

		const Tensor scalar = full(Shape{}, 1.0F);
		EXPECT_TRUE(scalar.shape().is_scalar());
		EXPECT_EQ(scalar.numel(), 1);

		const Tensor empty = full(Shape{ 2, 0, 3 }, 1.0F);
		EXPECT_EQ(empty.shape(), (Shape{ 2, 0, 3 }));
		EXPECT_EQ(empty.numel(), 0);
	}

	TEST(CreationTest, FullAndConstantConveniencesProduceValues)
	{
		const std::array<float, 4> full_values{ -2.0F, -2.0F, -2.0F, -2.0F };
		const std::array<float, 4> zero_values{};
		const std::array<float, 4> one_values{ 1.0F, 1.0F, 1.0F, 1.0F };

		expect_tensor_eq(full(Shape{ 2, 2 }, -2.0F), full_values);
		expect_tensor_eq(zeros(Shape{ 2, 2 }), zero_values);
		expect_tensor_eq(ones(Shape{ 2, 2 }), one_values);
	}

	TEST(CreationTest, LikeOperationsInheritSpecificationByDefault)
	{
		const RandomKey key{ 11 };
		const Tensor input = full(
			Shape{ 2, 3 }, 0.0F,
			TensorOptions{ DType::Float32, Device::cpu(4) });

		const std::array<Tensor, 5> outputs{
			full_like(input, 1.0F),
			zeros_like(input),
			ones_like(input),
			uniform_like(input, -1.0F, 1.0F, key),
			normal_like(input, 0.0F, 1.0F, key)
		};

		for (const Tensor& output : outputs)
		{
			EXPECT_EQ(output.shape(), input.shape());
			EXPECT_EQ(output.dtype(), input.dtype());
			EXPECT_EQ(output.device(), input.device());
		}
	}

	TEST(CreationTest, LikeOperationsHonorExplicitOptions)
	{
		const RandomKey key{ 13 };
		const Tensor input = full(
			Shape{ 2, 3 }, 0.0F,
			TensorOptions{ DType::Float32, Device::cpu(4) });
		const TensorOptions options{ DType::Float32, Device::cpu(2) };

		const std::array<Tensor, 5> outputs{
			full_like(input, 1.0F, options),
			zeros_like(input, options),
			ones_like(input, options),
			uniform_like(input, -1.0F, 1.0F, key, options),
			normal_like(input, 0.0F, 1.0F, key, options)
		};

		for (const Tensor& output : outputs)
		{
			EXPECT_EQ(output.shape(), input.shape());
			EXPECT_EQ(output.dtype(), options.dtype);
			EXPECT_EQ(output.device(), options.device);
		}
	}
}
