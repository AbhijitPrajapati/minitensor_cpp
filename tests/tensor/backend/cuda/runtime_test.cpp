#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "tensor/backend/cuda/cuda_buffer.hpp"
#include "tensor/backend/cuda/cuda_runtime.hpp"
#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/support/test_buffer.hpp"

namespace minitensor::test
{
using CudaRuntimeDeviceTest = CudaDeviceTest;

TEST(CudaRuntimeTest, RejectsNonCudaDevices)
{
    EXPECT_THROW((detail::cuda::CudaRuntime{Device::cpu()}), std::invalid_argument);
}

TEST(CudaRuntimeTest, RepresentsZeroByteAllocations)
{
    detail::cuda::CudaRuntime runtime;
    const detail::BufferRef buffer = runtime.allocate(0);
    ASSERT_NE(buffer, nullptr);
    const auto *cuda_buffer = dynamic_cast<const detail::cuda::CudaBuffer *>(buffer.get());
    ASSERT_NE(cuda_buffer, nullptr);
    EXPECT_EQ(runtime.device(), Device::cuda());
    EXPECT_EQ(buffer->device(), runtime.device());
    EXPECT_EQ(buffer->size_bytes(), 0);
    EXPECT_EQ(cuda_buffer->data(), nullptr);

    EXPECT_NO_THROW(runtime.copy_from_host(*buffer, 0, std::span<const std::byte>{}));
    EXPECT_NO_THROW(runtime.copy_to_host(std::span<std::byte>{}, *buffer, 0));
}

TEST(CudaRuntimeTest, RejectsForeignBufferKindsEvenForEmptyCopies)
{
    detail::cuda::CudaRuntime runtime;
    const detail::BufferRef buffer = make_test_buffer(0, Device::cuda());
    EXPECT_THROW(runtime.copy_from_host(*buffer, 0, std::span<const std::byte>{}), std::invalid_argument);
    EXPECT_THROW(runtime.copy_to_host(std::span<std::byte>{}, *buffer, 0), std::invalid_argument);
}

TEST(CudaRuntimeTest, RejectsBuffersFromAnotherCudaDeviceEvenForEmptyCopies)
{
    detail::cuda::CudaRuntime runtime;
    // Zero-byte buffers need no hardware allocation, so this validation test
    // also runs on machines with fewer than two GPUs.
    detail::cuda::CudaBuffer buffer{0, Device::cuda(1)};
    EXPECT_THROW(runtime.copy_from_host(buffer, 0, std::span<const std::byte>{}), std::invalid_argument);
    EXPECT_THROW(runtime.copy_to_host(std::span<std::byte>{}, buffer, 0), std::invalid_argument);
}

TEST_F(CudaRuntimeDeviceTest, AllocatesDeviceMemoryForItsDevice)
{
    detail::cuda::CudaRuntime runtime;
    const detail::BufferRef buffer = runtime.allocate(2 * sizeof(float));
    ASSERT_NE(buffer, nullptr);
    const auto *cuda_buffer = dynamic_cast<const detail::cuda::CudaBuffer *>(buffer.get());
    ASSERT_NE(cuda_buffer, nullptr);
    EXPECT_EQ(buffer->device(), runtime.device());
    EXPECT_EQ(buffer->size_bytes(), 2 * sizeof(float));
    ASSERT_NE(cuda_buffer->data(), nullptr);

    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, cuda_buffer->data()), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    EXPECT_EQ(attributes.device, 0);
}

TEST_F(CudaRuntimeDeviceTest, CopiesBytesAtExplicitOffsets)
{
    detail::cuda::CudaRuntime runtime;
    const detail::BufferRef buffer = runtime.allocate(6);
    const std::array<std::byte, 6> initial{};
    runtime.copy_from_host(*buffer, 0, initial);

    const std::array<std::byte, 3> source{std::byte{1}, std::byte{2}, std::byte{3}};
    runtime.copy_from_host(*buffer, 2, source);
    const std::array<std::byte, 6> expected{std::byte{0}, std::byte{0}, std::byte{1},
                                            std::byte{2}, std::byte{3}, std::byte{0}};
    std::array<std::byte, 6> result{};
    runtime.copy_to_host(result, *buffer, 0);
    EXPECT_EQ(result, expected);

    std::array<std::byte, 2> destination{};
    runtime.copy_to_host(destination, *buffer, 3);
    EXPECT_EQ(destination, (std::array<std::byte, 2>{std::byte{2}, std::byte{3}}));
}

TEST_F(CudaRuntimeDeviceTest, AcceptsEmptyCopiesAtTheBufferBoundary)
{
    detail::cuda::CudaRuntime runtime;
    const detail::BufferRef buffer = runtime.allocate(4);
    EXPECT_NO_THROW(runtime.copy_from_host(*buffer, 4, std::span<const std::byte>{}));
    EXPECT_NO_THROW(runtime.copy_to_host(std::span<std::byte>{}, *buffer, 4));
}

TEST_F(CudaRuntimeDeviceTest, ValidatesTransferBounds)
{
    detail::cuda::CudaRuntime runtime;
    const detail::BufferRef buffer = runtime.allocate(4);
    const std::array<std::byte, 2> source{std::byte{1}, std::byte{2}};
    std::array<std::byte, 2> destination{};
    EXPECT_THROW(runtime.copy_from_host(*buffer, 3, source), std::out_of_range);
    EXPECT_THROW(runtime.copy_to_host(destination, *buffer, 3), std::out_of_range);
    EXPECT_THROW(runtime.copy_from_host(*buffer, 4, source), std::out_of_range);
    EXPECT_THROW(runtime.copy_to_host(destination, *buffer, 4), std::out_of_range);

    const std::size_t excessive_offset = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(runtime.copy_from_host(*buffer, excessive_offset, std::span<const std::byte>{}), std::out_of_range);
    EXPECT_THROW(runtime.copy_to_host(std::span<std::byte>{}, *buffer, excessive_offset), std::out_of_range);
}
} // namespace minitensor::test
