#include <minitensor/types.hpp>

#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/core/axis.hpp"

namespace minitensor::test
{
	TEST(AxisTest, NormalizesPositiveAndNegativeAxes)
	{
		EXPECT_EQ(detail::normalize_axis(0, 3), 0);
		EXPECT_EQ(detail::normalize_axis(1, 3), 1);
		EXPECT_EQ(detail::normalize_axis(-1, 3), 2);
		EXPECT_EQ(detail::normalize_axis(-3, 3), 0);
	}

	TEST(AxisTest, RejectsAxesOutsideTheRank)
	{
		EXPECT_THROW((void)detail::normalize_axis(3, 3), std::out_of_range);
		EXPECT_THROW((void)detail::normalize_axis(-4, 3), std::out_of_range);
		EXPECT_THROW((void)detail::normalize_axis(0, 0), std::out_of_range);
	}

	TEST(AxisTest, RejectsRanksThatCannotBeRepresentedByAxis)
	{
		const auto oversized_rank = static_cast<Shape::size_type>(
			std::numeric_limits<Axis>::max()) + 1;
		EXPECT_THROW(
			(void)detail::normalize_axis(0, oversized_rank),
			std::overflow_error);
	}
}
