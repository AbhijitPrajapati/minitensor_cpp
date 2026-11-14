#pragma once

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace minitensor::test
{
// Hardware-dependent tests skip only when CUDA reports no usable driver/device.
// Other CUDA errors fail the test rather than hiding a broken environment.
class CudaDeviceTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        const cudaError_t status = cudaGetDeviceCount(&device_count_);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        {
            GTEST_SKIP() << cudaGetErrorString(status);
        }
        ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
        if (device_count_ == 0)
        {
            GTEST_SKIP() << "No CUDA devices are available";
        }

        ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);
        restore_device_ = true;
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    void TearDown() override
    {
        if (restore_device_)
        {
            EXPECT_EQ(cudaSetDevice(original_device_), cudaSuccess);
        }
    }

    int device_count_{0};

  private:
    int original_device_{0};
    bool restore_device_{false};
};
} // namespace minitensor::test
