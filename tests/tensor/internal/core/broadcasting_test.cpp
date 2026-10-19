#include <minitensor/types.hpp>

#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/core/broadcasting.hpp"

namespace minitensor::test
{
TEST(BroadcastShapeTest, CombinesAlignedAndLeadingDimensions)
{
    EXPECT_EQ(detail::broadcast_shape(Shape{2, 3}, Shape{2, 3}), (Shape{2, 3}));
    EXPECT_EQ(detail::broadcast_shape(Shape{2, 1, 4}, Shape{3, 4}), (Shape{2, 3, 4}));
    EXPECT_EQ(detail::broadcast_shape(Shape{}, Shape{2, 3}), (Shape{2, 3}));
    EXPECT_EQ(detail::broadcast_shape(Shape{2, 0, 3}, Shape{1, 3}), (Shape{2, 0, 3}));
}

TEST(BroadcastShapeTest, RejectsIncompatibleDimensions)
{
    EXPECT_THROW((void)detail::broadcast_shape(Shape{2, 3}, Shape{2, 2}), std::invalid_argument);
}
} // namespace minitensor::test
