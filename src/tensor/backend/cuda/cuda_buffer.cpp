#include "cuda_buffer.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>

#include "detail/cuda_check.hpp"

namespace minitensor::detail::cuda
{

// Explicit device management is used instead of the device guard for the low level buffer.

CudaBuffer::CudaBuffer(std::size_t size_bytes, Device device) : device_(device), size_bytes_(size_bytes)
{
    // Validate the device, even for empty buffers.
    if (device_.type() != DeviceType::Cuda)
    {
        throw std::invalid_argument{"cuda device is required"};
    }
    if (!std::in_range<int>(device_.index()))
    {
        throw std::invalid_argument{"CUDA device index is too large"};
    }
    if (size_bytes_ == 0)
    {
        return;
    }

    // Store the previous device.
    const int target_device = static_cast<int>(device_.index());
    int previous_device = 0;
    cuda_check(cudaGetDevice(&previous_device), "cudaGetDevice");

    try
    {
        // Set the device.
        cuda_check(cudaSetDevice(target_device), "cudaSetDevice");

        // Allocate device memory.
        void *allocation = nullptr;
        cuda_check(cudaMalloc(&allocation, size_bytes_), "cudaMalloc");
        data_ = static_cast<std::byte *>(allocation);

        // Restore the previous device.
        cuda_check(cudaSetDevice(previous_device), "restore CUDA device");
    }
    catch (...)
    {
        // A failed constructor will not run this buffer's destructor. Report errors without throwing.
        if (data_ != nullptr)
        {
            // Free remaining device memory.
            if (cuda_report(cudaSetDevice(target_device), "select CUDA device for constructor cleanup"))
            {
                cuda_report(cudaFree(data_), "free allocation after constructor failure");
            }
        }

        // Preserve the original exception even if cleanup or restoration fails.
        cuda_report(cudaSetDevice(previous_device), "restore CUDA device after constructor failure");
        throw;
    }
}

CudaBuffer::~CudaBuffer()
{
    if (data_ == nullptr)
    {
        return;
    }

    // Store the previous device.
    int previous_device = 0;
    if (!cuda_report(cudaGetDevice(&previous_device), "get CUDA device during buffer destruction"))
    {
        return;
    }

    // Attempt to set the target device. Report on failure.
    const int target_device = static_cast<int>(device_.index());
    if (!cuda_report(cudaSetDevice(target_device), "select CUDA device during buffer destruction"))
    {
        cuda_report(cudaSetDevice(previous_device), "restore CUDA device after selection failure");
        return;
    }

    // Free memory and reset device.
    cuda_report(cudaFree(data_), "cudaFree");
    cuda_report(cudaSetDevice(previous_device), "restore CUDA device during buffer destruction");
}

Device CudaBuffer::device() const noexcept
{
    return device_;
}

std::size_t CudaBuffer::size_bytes() const noexcept
{
    return size_bytes_;
}

std::byte *CudaBuffer::data() noexcept
{
    return data_;
}

const std::byte *CudaBuffer::data() const noexcept
{
    return data_;
}
} // namespace minitensor::detail::cuda
