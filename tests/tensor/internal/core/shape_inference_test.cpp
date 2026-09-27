#include <minitensor/types.hpp>

#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/core/shape_inference.hpp"
#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(BroadcastShapeTest, CombinesAlignedAndLeadingDimensions)
	{
		EXPECT_EQ(
			detail::broadcast_shape(Shape{ 2, 3 }, Shape{ 2, 3 }),
			(Shape{ 2, 3 }));
		EXPECT_EQ(
			detail::broadcast_shape(Shape{ 2, 1, 4 }, Shape{ 3, 4 }),
			(Shape{ 2, 3, 4 }));
		EXPECT_EQ(
			detail::broadcast_shape(Shape{}, Shape{ 2, 3 }),
			(Shape{ 2, 3 }));
		EXPECT_EQ(
			detail::broadcast_shape(Shape{ 2, 0, 3 }, Shape{ 1, 3 }),
			(Shape{ 2, 0, 3 }));
	}

	TEST(BroadcastShapeTest, RejectsIncompatibleDimensions)
	{
		EXPECT_THROW(
			(void)detail::broadcast_shape(Shape{ 2, 3 }, Shape{ 2, 2 }),
			std::invalid_argument);
	}

	TEST(MatmulShapeTest, HandlesVectorAndMatrixPromotion)
	{
		EXPECT_EQ(detail::matmul_output_shape(Shape{ 3 }, Shape{ 3 }), Shape{});
		EXPECT_EQ(
			detail::matmul_output_shape(Shape{ 2, 3 }, Shape{ 3 }),
			(Shape{ 2 }));
		EXPECT_EQ(
			detail::matmul_output_shape(Shape{ 3 }, Shape{ 3, 4 }),
			(Shape{ 4 }));
		EXPECT_EQ(
			detail::matmul_output_shape(Shape{ 2, 3 }, Shape{ 3, 4 }),
			(Shape{ 2, 4 }));
	}

	TEST(MatmulShapeTest, BroadcastsBatchDimensions)
	{
		EXPECT_EQ(
			detail::matmul_output_shape(
				Shape{ 2, 1, 3, 4 }, Shape{ 5, 4, 6 }),
			(Shape{ 2, 5, 3, 6 }));
		EXPECT_EQ(
			detail::matmul_output_shape(
				Shape{ 4 }, Shape{ 2, 5, 4, 6 }),
			(Shape{ 2, 5, 6 }));
		EXPECT_EQ(
			detail::matmul_output_shape(
				Shape{ 2, 5, 3, 4 }, Shape{ 4 }),
			(Shape{ 2, 5, 3 }));
	}

	TEST(MatmulShapeTest, ValidatesRanksContractionsAndBatches)
	{
		EXPECT_THROW(
			(void)detail::matmul_output_shape(Shape{}, Shape{ 2, 2 }),
			std::invalid_argument);
		EXPECT_THROW(
			(void)detail::matmul_output_shape(Shape{ 2, 3 }, Shape{ 2, 4 }),
			std::invalid_argument);
		EXPECT_THROW(
			(void)detail::matmul_output_shape(
				Shape{ 2, 3, 4 }, Shape{ 5, 4, 6 }),
			std::invalid_argument);
	}
}
