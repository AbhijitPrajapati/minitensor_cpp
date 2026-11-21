#include <minitensor/random.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
using CudaCreationKernelTest = CudaDeviceTest;

TEST_F(CudaCreationKernelTest, FullWritesOffsetAndEmptyOutputs)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{2, 3}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(8);
    const std::array<float, 8> initial{-10, -10, -10, -10, -10, -10, -10, -10};
    harness.copy_from_host<float>(buffer, initial);
    const detail::Materialization storage{buffer, detail::Layout{{3, 1}, 1}};
    const detail::MutableTensorView output{spec, storage};
    const detail::FullPrimitive primitive{spec, 2.5F};
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);

    EXPECT_EQ(harness.copy_to_host<float>(buffer), (std::vector<float>{-10, 2.5F, 2.5F, 2.5F, 2.5F, 2.5F, 2.5F, -10}));

    const detail::TensorSpec empty_spec{Shape{2, 0}, DType::Float32, Device::cuda()};
    const detail::BufferRef empty_buffer = harness.allocate<float>(0);
    const detail::Materialization empty_storage{empty_buffer, detail::Layout::contiguous(empty_spec.shape)};
    const detail::MutableTensorView empty_output{empty_spec, empty_storage};
    const detail::FullPrimitive empty_primitive{empty_spec, 4.0F};
    EXPECT_NO_THROW(harness.invoke(empty_primitive, std::span<const detail::TensorView>{}, empty_output));
}

TEST_F(CudaCreationKernelTest, FullPreservesInt64ValuesAndOutputOffsets)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{3}, DType::Int64, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<std::int64_t>(5);
    const std::array<std::int64_t, 5> initial{-10, -10, -10, -10, -10};
    harness.copy_from_host<std::int64_t>(buffer, initial);
    const detail::Materialization storage{buffer, detail::Layout{{1}, 1}};
    const detail::MutableTensorView output{spec, storage};
    constexpr std::int64_t value = 9'007'199'254'740'993;
    const detail::FullPrimitive primitive{spec, value};
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);

    EXPECT_EQ(harness.copy_to_host<std::int64_t>(buffer), (std::vector<std::int64_t>{-10, value, value, value, -10}));
}

TEST_F(CudaCreationKernelTest, FullWritesScalarOutputs)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(3);
    const std::array<float, 3> initial{-10, -10, -10};
    harness.copy_from_host<float>(buffer, initial);
    const detail::Materialization storage{buffer, detail::Layout::contiguous(spec.shape, 1)};
    const detail::MutableTensorView output{spec, storage};
    const detail::FullPrimitive primitive{spec, -3.5F};
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);

    EXPECT_EQ(harness.copy_to_host<float>(buffer), (std::vector<float>{-10, -3.5F, -10}));
}

TEST_F(CudaCreationKernelTest, FullWritesElementCountsThatAreNotMultiplesOfTheBlockSize)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{1003}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(1005);
    const std::vector<float> initial(1005, -10.0F);
    harness.copy_from_host<float>(buffer, initial);
    const detail::Materialization storage{buffer, detail::Layout::contiguous(spec.shape, 1)};
    const detail::MutableTensorView output{spec, storage};
    const detail::FullPrimitive primitive{spec, 2.5F};
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);

    std::vector<float> expected(1005, 2.5F);
    expected.front() = -10.0F;
    expected.back() = -10.0F;
    EXPECT_EQ(harness.copy_to_host<float>(buffer), expected);
}

TEST_F(CudaCreationKernelTest, UniformIsDeterministicAndHonorsOutputOffsetAndKey)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{4099}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(4101);
    const std::vector<float> initial(4101, -100.0F);
    harness.copy_from_host<float>(buffer, initial);
    const detail::Materialization storage{buffer, detail::Layout::contiguous(spec.shape, 1)};
    const detail::MutableTensorView output{spec, storage};
    const detail::UniformPrimitive primitive{spec, -2.0F, 3.0F, RandomKey{101}};

    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    const auto first = harness.copy_to_host<float>(buffer);
    EXPECT_FLOAT_EQ(first.front(), -100.0F);
    EXPECT_FLOAT_EQ(first.back(), -100.0F);
    EXPECT_TRUE(
        std::all_of(first.begin() + 1, first.end() - 1, [](float value) { return value >= -2.0F && value < 3.0F; }));

    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    EXPECT_EQ(harness.copy_to_host<float>(buffer), first);

    const detail::UniformPrimitive distinct{spec, -2.0F, 3.0F, RandomKey{102}};
    harness.invoke(distinct, std::span<const detail::TensorView>{}, output);
    EXPECT_NE(harness.copy_to_host<float>(buffer), first);
}

TEST_F(CudaCreationKernelTest, UniformPreservesHalfOpenBoundsForNarrowAndExtremeRanges)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{1003}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(1003);
    const detail::Materialization storage{buffer, detail::Layout::contiguous(spec.shape)};
    const detail::MutableTensorView output{spec, storage};

    const float low = 1.0F;
    const float high = std::nextafter(low, 2.0F);
    const detail::UniformPrimitive narrow{spec, low, high, RandomKey{201}};
    harness.invoke(narrow, std::span<const detail::TensorView>{}, output);
    EXPECT_EQ(harness.copy_to_host<float>(buffer), std::vector<float>(1003, low));

    const float limit = std::numeric_limits<float>::max();
    const detail::UniformPrimitive wide{spec, -limit, limit, RandomKey{202}};
    harness.invoke(wide, std::span<const detail::TensorView>{}, output);
    const auto values = harness.copy_to_host<float>(buffer);
    EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                            [limit](float value) { return std::isfinite(value) && value >= -limit && value < limit; }));
}

TEST_F(CudaCreationKernelTest, UniformSupportsScalarsEmptyOutputsAndEqualBounds)
{
    CudaKernelHarness harness;
    for (const Shape &shape : {Shape{}, Shape{2, 0}, Shape{1003}})
    {
        SCOPED_TRACE(shape.numel());
        const auto count = shape.numel();
        const detail::TensorSpec spec{shape, DType::Float32, Device::cuda()};
        const detail::BufferRef buffer = harness.allocate<float>(count + 2);
        const std::vector<float> initial(count + 2, -100.0F);
        harness.copy_from_host<float>(buffer, initial);
        const detail::Materialization storage{buffer, detail::Layout::contiguous(shape, 1)};
        const detail::MutableTensorView output{spec, storage};

        const detail::UniformPrimitive primitive{spec, 0.0F, 1.0F, RandomKey{301}};
        harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
        const auto values = harness.copy_to_host<float>(buffer);
        EXPECT_FLOAT_EQ(values.front(), -100.0F);
        EXPECT_FLOAT_EQ(values.back(), -100.0F);
        EXPECT_TRUE(std::all_of(values.begin() + 1, values.end() - 1,
                                [](float value) { return value >= 0.0F && value < 1.0F; }));

        const detail::UniformPrimitive constant{spec, 3.5F, 3.5F, RandomKey{302}};
        harness.invoke(constant, std::span<const detail::TensorView>{}, output);
        std::vector<float> expected(count + 2, 3.5F);
        expected.front() = -100.0F;
        expected.back() = -100.0F;
        EXPECT_EQ(harness.copy_to_host<float>(buffer), expected);
    }
}

TEST_F(CudaCreationKernelTest, NormalIsDeterministicAndHonorsOutputOffsetAndKey)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{4099}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(4101);
    const std::vector<float> initial(4101, -100.0F);
    harness.copy_from_host<float>(buffer, initial);
    const detail::Materialization storage{buffer, detail::Layout::contiguous(spec.shape, 1)};
    const detail::MutableTensorView output{spec, storage};
    const detail::NormalPrimitive primitive{spec, 1.5F, 2.0F, RandomKey{401}};

    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    const auto first = harness.copy_to_host<float>(buffer);
    EXPECT_FLOAT_EQ(first.front(), -100.0F);
    EXPECT_FLOAT_EQ(first.back(), -100.0F);
    EXPECT_TRUE(std::all_of(first.begin() + 1, first.end() - 1,
                            [](float value) { return std::isfinite(value) && value != -100.0F; }));

    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    EXPECT_EQ(harness.copy_to_host<float>(buffer), first);

    const detail::NormalPrimitive distinct{spec, 1.5F, 2.0F, RandomKey{402}};
    harness.invoke(distinct, std::span<const detail::TensorView>{}, output);
    EXPECT_NE(harness.copy_to_host<float>(buffer), first);
}

TEST_F(CudaCreationKernelTest, NormalSupportsScalarsEmptyOutputsAndZeroStandardDeviation)
{
    CudaKernelHarness harness;
    for (const Shape &shape : {Shape{}, Shape{2, 0}, Shape{1003}})
    {
        SCOPED_TRACE(shape.numel());
        const auto count = shape.numel();
        const detail::TensorSpec spec{shape, DType::Float32, Device::cuda()};
        const detail::BufferRef buffer = harness.allocate<float>(count + 2);
        const std::vector<float> initial(count + 2, -100.0F);
        harness.copy_from_host<float>(buffer, initial);
        const detail::Materialization storage{buffer, detail::Layout::contiguous(shape, 1)};
        const detail::MutableTensorView output{spec, storage};

        const detail::NormalPrimitive primitive{spec, 0.0F, 1.0F, RandomKey{501}};
        harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
        const auto values = harness.copy_to_host<float>(buffer);
        EXPECT_FLOAT_EQ(values.front(), -100.0F);
        EXPECT_FLOAT_EQ(values.back(), -100.0F);
        EXPECT_TRUE(std::all_of(values.begin() + 1, values.end() - 1,
                                [](float value) { return std::isfinite(value) && value != -100.0F; }));

        const detail::NormalPrimitive constant{spec, -3.5F, 0.0F, RandomKey{502}};
        harness.invoke(constant, std::span<const detail::TensorView>{}, output);
        std::vector<float> expected(count + 2, -3.5F);
        expected.front() = -100.0F;
        expected.back() = -100.0F;
        EXPECT_EQ(harness.copy_to_host<float>(buffer), expected);
    }
}

TEST_F(CudaCreationKernelTest, NormalHasExpectedMeanAndVariance)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{16387}, DType::Float32, Device::cuda()};
    const detail::BufferRef buffer = harness.allocate<float>(spec.shape.numel());
    const detail::Materialization storage{buffer, detail::Layout::contiguous(spec.shape)};
    const detail::MutableTensorView output{spec, storage};
    const detail::NormalPrimitive primitive{spec, 1.5F, 2.0F, RandomKey{601}};
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    const auto values = harness.copy_to_host<float>(buffer);

    double sum = 0.0;
    double square_sum = 0.0;
    for (const float value : values)
    {
        ASSERT_TRUE(std::isfinite(value));
        const double sample = value;
        sum += sample;
        square_sum += sample * sample;
    }
    const double mean = sum / static_cast<double>(values.size());
    const double variance = square_sum / static_cast<double>(values.size()) - mean * mean;
    EXPECT_NEAR(mean, 1.5, 0.1);
    EXPECT_NEAR(variance, 4.0, 0.25);
}
} // namespace minitensor::test
