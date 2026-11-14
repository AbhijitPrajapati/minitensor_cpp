#include "cuda_buffer.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>

#include "detail/safety.hpp"

namespace minitensor::detail::cuda
{

CudaBuffer::CudaBuffer(std::size_t size_bytes, Device device) : device_(device), size_bytes_(size_bytes)
{
    if (device_.type() != DeviceType::Cuda)
    {
        throw std::invalid_argument{"cuda device is required"};
    }

    // Device index must be a valid CUDA device index.
    if (!std::in_range<int>(device_.index()))
    {
        throw std::invalid_argument{"CUDA device index is too large"};
    }

    if (size_bytes_ == 0)
    {
        return;
    }

    const int target_device = static_cast<int>(device_.index());

    // Store the index of the previous device.
    int previous_device = 0;
    cuda_check(cudaGetDevice(&previous_device), "cudaGetDevice");

    try
    {
        // Set the buffer device.
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
        // The destructor will not run if construction fails.
        // Release any allocation successfully acquired above.
        // Use cuda_report to preserve the existing exception rather than throwing.
        if (data_ != nullptr)
        {
            if (cuda_report(cudaSetDevice(target_device), "select CUDA device for constructor cleanup"))
            {
                cuda_report(cudaFree(data_), "free allocation after constructor failure");
            }
        }

        // Attempt to restore the previous device.
        cuda_report(cudaSetDevice(previous_device), "restore CUDA device after constructor failure");

        // Preserve the original exception.
        throw;
    }
}

CudaBuffer::~CudaBuffer()
{
    if (data_ == nullptr)
    {
        return;
    }

    // Store the index of the previous device.
    int previous_device = 0;
    if (!cuda_report(cudaGetDevice(&previous_device), "get CUDA device during buffer destruction"))
    {
        // Return on failure.
        return;
    }

    const int target_device = static_cast<int>(device_.index());

    // Set the target device.
    if (!cuda_report(cudaSetDevice(target_device), "select CUDA device during buffer destruction"))
    {
        // Attempt to leave the caller's device selection unchanged on failure.
        cuda_report(cudaSetDevice(previous_device), "restore CUDA device after selection failure");
        return;
    }

    // Free device memory. Use cuda_report to continue restoration on failure.
    cuda_report(cudaFree(data_), "cudaFree");

    // Restore previous device.
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