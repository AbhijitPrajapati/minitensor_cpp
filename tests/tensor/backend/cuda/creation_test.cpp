#include <minitensor/types.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/creation/full.hpp"
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
} // namespace minitensor::test
