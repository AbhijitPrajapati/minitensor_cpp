#include <minitensor/types.hpp>

#include <array>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/core/broadcasting.hpp"

namespace minitensor::test
{
TEST(BroadcastExtentsTest, CombinesAlignedAndLeadingDimensions)
{
    const std::array<Extent, 2> equal{2, 3};
    const std::array<Extent, 3> lhs{2, 1, 4};
    const std::array<Extent, 2> rhs{3, 4};

    EXPECT_EQ(detail::broadcast_extents(equal, equal), (std::vector<Extent>{2, 3}));
    EXPECT_EQ(detail::broadcast_extents(lhs, rhs), (std::vector<Extent>{2, 3, 4}));
}

TEST(BroadcastExtentsTest, HandlesScalarsAndZeroExtents)
{
    const std::array<Extent, 0> scalar{};
    const std::array<Extent, 3> shape{2, 0, 3};
    const std::array<Extent, 2> broadcast{1, 3};

    EXPECT_EQ(detail::broadcast_extents(scalar, shape), (std::vector<Extent>{2, 0, 3}));
    EXPECT_EQ(detail::broadcast_extents(shape, broadcast), (std::vector<Extent>{2, 0, 3}));
}

TEST(BroadcastExtentsTest, RejectsIncompatibleDimensions)
{
    const std::array<Extent, 2> lhs{2, 3};
    const std::array<Extent, 2> rhs{2, 2};
    EXPECT_THROW((void)detail::broadcast_extents(lhs, rhs), std::invalid_argument);
}
} // namespace minitensor::test
