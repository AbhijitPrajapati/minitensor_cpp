#include <minitensor/types.hpp>

#include <array>

#include <gtest/gtest.h>

#include "tensor/core/reduction.hpp"

namespace minitensor::test
{
TEST(ReductionShapeTest, RemovesSelectedDimensionsAndPreservesTheirOrder)
{
    const Shape input{2, 3, 4, 5};
    const std::array<Shape::size_type, 1> first{0};
    const std::array<Shape::size_type, 1> middle{1};
    const std::array<Shape::size_type, 1> last{3};
    const std::array<Shape::size_type, 2> separated{0, 2};

    EXPECT_EQ(detail::reduction_shape(input, first, false), (Shape{3, 4, 5}));
    EXPECT_EQ(detail::reduction_shape(input, middle, false), (Shape{2, 4, 5}));
    EXPECT_EQ(detail::reduction_shape(input, last, false), (Shape{2, 3, 4}));
    EXPECT_EQ(detail::reduction_shape(input, separated, false), (Shape{3, 5}));
}

TEST(ReductionShapeTest, KeepsReducedDimensionsAsSingletons)
{
    const Shape input{2, 3, 4, 5};
    const std::array<Shape::size_type, 2> axes{1, 3};

    EXPECT_EQ(detail::reduction_shape(input, axes, true), (Shape{2, 1, 4, 1}));
    EXPECT_EQ(input, (Shape{2, 3, 4, 5}));
}

TEST(ReductionShapeTest, ReducesAllDimensionsToScalarOrSingletons)
{
    const std::array<Shape::size_type, 3> axes{0, 1, 2};

    EXPECT_EQ(detail::reduction_shape(Shape{2, 3, 4}, axes, false), Shape{});
    EXPECT_EQ(detail::reduction_shape(Shape{2, 3, 4}, axes, true), (Shape{1, 1, 1}));
}

TEST(ReductionShapeTest, PreservesShapesWhenNoAxesAreReduced)
{
    const std::array<Shape::size_type, 0> axes{};

    EXPECT_EQ(detail::reduction_shape(Shape{2, 1, 3}, axes, false), (Shape{2, 1, 3}));
    EXPECT_EQ(detail::reduction_shape(Shape{2, 1, 3}, axes, true), (Shape{2, 1, 3}));
    EXPECT_EQ(detail::reduction_shape(Shape{}, axes, false), Shape{});
    EXPECT_EQ(detail::reduction_shape(Shape{}, axes, true), Shape{});
}

TEST(ReductionShapeTest, HandlesZeroAndSingletonExtents)
{
    const Shape input{2, 0, 1, 3};
    const std::array<Shape::size_type, 1> empty_axis{1};
    const std::array<Shape::size_type, 1> singleton_axis{2};

    EXPECT_EQ(detail::reduction_shape(input, empty_axis, false), (Shape{2, 1, 3}));
    EXPECT_EQ(detail::reduction_shape(input, empty_axis, true), (Shape{2, 1, 1, 3}));
    EXPECT_EQ(detail::reduction_shape(input, singleton_axis, false), (Shape{2, 0, 3}));
    EXPECT_EQ(detail::reduction_shape(input, singleton_axis, true), input);
}
} // namespace minitensor::test
