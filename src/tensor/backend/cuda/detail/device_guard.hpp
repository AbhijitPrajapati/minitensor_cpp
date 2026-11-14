#pragma once

#include <stdexcept>
#include <utility>

#include <cuda_runtime_api.h>
#include <minitensor/types.hpp>

#include "cuda_check.hpp"

namespace minitensor::detail::cuda
{
class DeviceGuard final
{
  public:
    explicit DeviceGuard(Device device)
    {
        if (device.type() != DeviceType::Cuda)
        {
            throw std::invalid_argument{"DeviceGuard requires a CUDA device"};
        }

        if (!std::in_range<int>(device.index()))
        {
            throw std::invalid_argument{"CUDA device index is too large"};
        }

        // Store the previous device.
        cuda_check(cudaGetDevice(&previous_device_), "get current CUDA device");

        // Set the target device.
        try
        {
            cuda_check(cudaSetDevice(static_cast<int>(device.index())), "select CUDA device");
        }
        catch (...)
        {
            // A failed constructor will not the destructor.
            // Attempt restoration without replacing the exception.
            cuda_report(cudaSetDevice(previous_device_), "restore CUDA device after selection failure");
            throw;
        }

        active_ = true;
    }

    ~DeviceGuard() noexcept
    {
        if (active_)
        {
            // Preserve the original exception during unwinding.
            cuda_report(cudaSetDevice(previous_device_), "restore CUDA device during guard destruction");
        }
    }

    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    DeviceGuard(DeviceGuard &&) = delete;
    DeviceGuard &operator=(DeviceGuard &&) = delete;

    void restore()
    {
        if (!active_)
        {
            return;
        }

        // Reset the device.
        cuda_check(cudaSetDevice(previous_device_), "restore CUDA device");

        // Disable destructor restoration only after successful restoration.
        active_ = false;
    }

  private:
    int previous_device_{0};
    bool active_{false};
};
} // namespace minitensor::detail::cuda