#include "cuda_runtime.hpp"

#include <memory>

#include <cuda_runtime_api.h>
#include <minitensor/types.hpp>

#include "cuda_buffer.hpp"
#include "detail/safety.hpp"

namespace minitensor::detail::cuda
{
CudaRuntime::CudaRuntime(Device device) : device_{device}
{
    if (device_.type() != DeviceType::Cuda)
    {
        throw std::invalid_argument{"CudaRuntime requires a CUDA device"};
    }
}

Device CudaRuntime::device() const noexcept
{
    return device_;
}

BufferRef CudaRuntime::allocate(std::size_t size_bytes)
{
    return std::make_shared<CudaBuffer>(size_bytes, device_);
}

void CudaRuntime::copy_from_host(Buffer &destination, std::size_t destination_offset_bytes,
                                 std::span<const std::byte> source)
{
    auto *buffer = dynamic_cast<CudaBuffer *>(&destination);
    if (!buffer)
    {
        throw std::invalid_argument{"CudaRuntime requires a CudaBuffer destination"};
    }
    if (buffer->device() != device())
    {
        throw std::invalid_argument{"destination buffer belongs to a different device"};
    }

    const std::size_t buffer_size = buffer->size_bytes();
    if (destination_offset_bytes > buffer_size || source.size() > buffer_size - destination_offset_bytes)
    {
        throw std::out_of_range{"host-to-device copy exceeds destination buffer"};
    }

    if (source.empty())
    {
        return;
    }
    cuda_check(
        cudaMemcpy(buffer->data() + destination_offset_bytes, source.data(), source.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy to device");
}

void CudaRuntime::copy_to_host(std::span<std::byte> destination, const Buffer &source, std::size_t source_offset_bytes)
{
    const auto *buffer = dynamic_cast<const CudaBuffer *>(&source);
    if (!buffer)
    {
        throw std::invalid_argument{"CudaRuntime requires a CudaBuffer source"};
    }
    if (buffer->device() != device())
    {
        throw std::invalid_argument{"source buffer belongs to a different device"};
    }

    const std::size_t buffer_size = buffer->size_bytes();
    if (source_offset_bytes > buffer_size || destination.size() > buffer_size - source_offset_bytes)
    {
        throw std::out_of_range{"device-to-host copy exceeds source buffer"};
    }

    if (destination.empty())
    {
        return;
    }
    cuda_check(cudaMemcpy(destination.data(), buffer->data() + source_offset_bytes, destination.size(),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy to host");
}
} // namespace minitensor::detail::cuda