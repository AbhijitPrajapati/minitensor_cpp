#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <span>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/tensor_access.hpp"
#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(ReduceToShapeTest, ReusesMatchingShapes)
	{
		const Tensor input = full(Shape{ 2, 3 }, 1.0F);
		const Tensor result = detail::reduce_to_shape(input, input.shape());
		EXPECT_EQ(detail::TensorAccess::value(result), detail::TensorAccess::value(input));
	}

	TEST(ReduceToShapeTest, ReducesLeadingAndAlignedDimensions)
	{
		const Tensor matrix = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });

		const Tensor leading = detail::reduce_to_shape(matrix, Shape{ 3 });
		const std::array<float, 3> columns{ 5.0F, 7.0F, 9.0F };
		expect_tensor_eq(leading, columns);

		const Tensor aligned = detail::reduce_to_shape(matrix, Shape{ 2, 1 });
		const std::array<float, 2> rows{ 6.0F, 15.0F };
		EXPECT_EQ(aligned.shape(), (Shape{ 2, 1 }));
		expect_tensor_eq(aligned, rows);
		EXPECT_EQ(aligned.dtype(), matrix.dtype());
		EXPECT_EQ(aligned.device(), matrix.device());
	}

	TEST(ReduceToShapeTest, CombinesMultipleReductionKinds)
	{
		const Tensor volume = from_data(
			{
				1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
				7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F
			},
			Shape{ 2, 2, 3 });
		const Tensor result = detail::reduce_to_shape(volume, Shape{ 1, 3 });
		const std::array<float, 3> expected{ 22.0F, 26.0F, 30.0F };
		EXPECT_EQ(result.shape(), (Shape{ 1, 3 }));
		expect_tensor_eq(result, expected);
	}

	TEST(ReduceToShapeTest, RemovesLeadingSingletonsAndReducesToScalar)
	{
		const Tensor input = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 1, 2, 3 });
		const Tensor squeezed = detail::reduce_to_shape(input, Shape{ 2, 3 });
		const std::array<float, 6> expected{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		expect_tensor_eq(squeezed, expected);
		EXPECT_FLOAT_EQ(item(detail::reduce_to_shape(input, Shape{})), 21.0F);
	}

	TEST(ReduceToShapeTest, UsesZeroForEmptyReductions)
	{
		const Tensor input = from_data(std::span<const float>{}, Shape{ 0, 3 });
		const Tensor result = detail::reduce_to_shape(input, Shape{ 1, 3 });
		const std::array<float, 3> expected{};
		expect_tensor_eq(result, expected);
	}

	TEST(ReduceToShapeTest, ValidatesTargetShape)
	{
		const Tensor input = full(Shape{ 2, 3 }, 1.0F);
		EXPECT_THROW(
			(void)detail::reduce_to_shape(input, Shape{ 1, 2, 3 }),
			std::invalid_argument);
		EXPECT_THROW(
			(void)detail::reduce_to_shape(input, Shape{ 2, 2 }),
			std::invalid_argument);
	}
}
