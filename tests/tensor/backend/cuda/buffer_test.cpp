#include <minitensor/types.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "tensor/backend/cuda/cuda_buffer.hpp"
#include "tensor/backend/cuda/support/device_test.hpp"

namespace minitensor::test
{
TEST(CudaBufferTest, RejectsNonCudaDevices)
{
    EXPECT_THROW((detail::cuda::CudaBuffer{0, Device::cpu()}), std::invalid_argument);
}

TEST(CudaBufferTest, RejectsDeviceIndicesThatDoNotFitCudaOrdinals)
{
    EXPECT_THROW((detail::cuda::CudaBuffer{0, Device::cuda(std::numeric_limits<std::uint32_t>::max())}),
                 std::invalid_argument);
}

TEST(CudaBufferTest, RepresentsAnEmptyBufferWithoutDeviceAllocation)
{
    detail::cuda::CudaBuffer buffer{0};
    const detail::cuda::CudaBuffer &const_buffer = buffer;
    EXPECT_EQ(buffer.device(), Device::cuda());
    EXPECT_EQ(buffer.size_bytes(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(const_buffer.data(), nullptr);
}

TEST_F(CudaDeviceTest, RejectsUnavailableDevicesAndRestoresTheCurrentDevice)
{
    // Valid ordinals end at device_count_ - 1.
    EXPECT_THROW((detail::cuda::CudaBuffer{4, Device::cuda(static_cast<std::uint32_t>(device_count_))}),
                 std::runtime_error);

    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, 0);
}

TEST_F(CudaDeviceTest, RestoresTheCurrentDeviceAfterAllocationAndDestruction)
{
    if (device_count_ < 2)
    {
        GTEST_SKIP() << "Device restoration requires two CUDA devices";
    }

    {
        detail::cuda::CudaBuffer buffer{4, Device::cuda(1)};
        const detail::cuda::CudaBuffer &const_buffer = buffer;
        EXPECT_EQ(buffer.device(), Device::cuda(1));
        EXPECT_EQ(buffer.size_bytes(), 4);
        EXPECT_EQ(buffer.data(), const_buffer.data());

        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, buffer.data()), cudaSuccess);
        EXPECT_EQ(attributes.device, 1);

        int current_device = -1;
        ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
        EXPECT_EQ(current_device, 0);
    }

    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, 0);
}
} // namespace minitensor::test
