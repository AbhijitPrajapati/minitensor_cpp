#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
static_assert(std::ranges::random_access_range<Shape>);
static_assert(std::ranges::random_access_range<const Shape>);
static_assert(std::same_as<std::ranges::range_reference_t<const Shape>, const Extent &>);

TEST(ShapeTest, RepresentsScalars)
{
    const Shape shape;

    EXPECT_EQ(shape.rank(), 0);
    EXPECT_TRUE(shape.is_scalar());
    EXPECT_TRUE(std::ranges::empty(shape));
    EXPECT_EQ(shape.numel(), 1);
}

TEST(ShapeTest, PreservesDimensionsAndCountsElements)
{
    const Shape from_list{2, 3, 4};
    const std::array<Extent, 3> expected{2, 3, 4};
    EXPECT_EQ(from_list.rank(), expected.size());
    EXPECT_TRUE(std::ranges::equal(from_list, expected));
    EXPECT_EQ(from_list[1], 3);
    EXPECT_EQ(from_list.numel(), 24);

    const std::vector<Extent> dimensions{5, 2, 3};
    const Shape from_vector{dimensions};
    EXPECT_TRUE(std::ranges::equal(from_vector, dimensions));
    EXPECT_EQ(from_vector.numel(), 30);
}

TEST(ShapeTest, HandlesEmptyShapesWithoutSpuriousOverflow)
{
    const Shape shape{2, 0, std::numeric_limits<Extent>::max()};
    EXPECT_EQ(shape.numel(), 0);
    EXPECT_FALSE(shape.is_scalar());
}

TEST(ShapeTest, ComparesDimensionSequences)
{
    EXPECT_EQ((Shape{2, 3}), (Shape{2, 3}));
    EXPECT_NE((Shape{2, 3}), (Shape{3, 2}));
}

TEST(ShapeTest, RejectsNegativeExtents)
{
    EXPECT_THROW((void)(Shape{3, -2, 1}), std::invalid_argument);
    EXPECT_THROW((void)(Shape{std::vector<Extent>{1, -1}}), std::invalid_argument);
}

TEST(ShapeTest, RejectsElementCountOverflow)
{
    EXPECT_THROW((void)(Shape{std::numeric_limits<Extent>::max(), 3}), std::overflow_error);
}

TEST(DTypeTest, ExposesFloat32Properties)
{
    EXPECT_EQ(dtype_size(DType::Float32), sizeof(float));
    EXPECT_EQ(dtype_name(DType::Float32), "float32");
    EXPECT_TRUE(is_floating_point(DType::Float32));
    EXPECT_FALSE(is_integral(DType::Float32));
    EXPECT_EQ(dtype_of_v<float>, DType::Float32);
}

TEST(DTypeTest, ExposesInt64Properties)
{
    EXPECT_EQ(dtype_size(DType::Int64), sizeof(std::int64_t));
    EXPECT_EQ(dtype_name(DType::Int64), "int64");
    EXPECT_FALSE(is_floating_point(DType::Int64));
    EXPECT_TRUE(is_integral(DType::Int64));
    EXPECT_EQ(dtype_of_v<std::int64_t>, DType::Int64);
}

TEST(DTypeTest, RejectsUnknownValues)
{
    const auto unknown = static_cast<DType>(255);
    EXPECT_THROW((void)dtype_size(unknown), std::invalid_argument);
    EXPECT_THROW((void)dtype_name(unknown), std::invalid_argument);
}

TEST(DeviceTest, DefaultsToCpuZeroAndPreservesIndices)
{
    const Device default_device;
    EXPECT_EQ(default_device, Device::cpu());
    EXPECT_EQ(default_device.type(), DeviceType::Cpu);
    EXPECT_EQ(default_device.index(), 0);

    const Device indexed = Device::cpu(3);
    EXPECT_EQ(indexed.type(), DeviceType::Cpu);
    EXPECT_EQ(indexed.index(), 3);
    EXPECT_NE(indexed, default_device);
}

TEST(TensorOptionsTest, HasValueSemantics)
{
    const TensorOptions defaults;
    EXPECT_EQ(defaults.dtype, DType::Float32);
    EXPECT_EQ(defaults.device, Device::cpu());

    const TensorOptions customized{DType::Float32, Device::cpu(4)};
    EXPECT_EQ(customized, (TensorOptions{DType::Float32, Device::cpu(4)}));
    EXPECT_NE(customized, defaults);
}
} // namespace minitensor::test
