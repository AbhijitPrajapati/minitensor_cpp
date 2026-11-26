#include "cuda_runtime.hpp"

#include <memory>

#include <cuda_runtime_api.h>
#include <minitensor/types.hpp>

#include "cuda_buffer.hpp"
#include "detail/cuda_check.hpp"
#include "detail/device_guard.hpp"
#include "tensor/backend/cpu/cpu_buffer.hpp"

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
    DeviceGuard guard(device());
    cuda_check(cudaMemcpyAsync(buffer->data() + destination_offset_bytes, source.data(), source.size(),
                               cudaMemcpyHostToDevice, nullptr),
               "cudaMemcpy to device");
    cuda_check(cudaStreamSynchronize(nullptr), "complete host-to-device copy");
    guard.restore();
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
    DeviceGuard guard(device());
    cuda_check(cudaMemcpyAsync(destination.data(), buffer->data() + source_offset_bytes, destination.size(),
                               cudaMemcpyDeviceToHost, nullptr),
               "cudaMemcpy to host");
    cuda_check(cudaStreamSynchronize(nullptr), "complete device-to-host copy");
    guard.restore();
}

void CudaRuntime::copy_buffer(Buffer &destination, std::size_t destination_offset_bytes, const Buffer &source,
                              std::size_t source_offset_bytes, std::size_t byte_count)
{
    // CUDA runtime copying supports CPU buffers but requires at least one CUDA endpoint.
    auto *target_device = dynamic_cast<CudaBuffer *>(&destination);
    const auto *source_device = dynamic_cast<const CudaBuffer *>(&source);
    auto *target_host = dynamic_cast<cpu::CpuBuffer *>(&destination);
    const auto *source_host = dynamic_cast<const cpu::CpuBuffer *>(&source);
    if ((!target_device && !target_host) || (!source_device && !source_host) || (!target_device && !source_device))
    {
        throw std::invalid_argument{"CudaRuntime requires CPU/CUDA buffers with at least one CUDA endpoint"};
    }
    if ((!target_device || target_device->device() != device()) &&
        (!source_device || source_device->device() != device()))
    {
        throw std::invalid_argument{"neither CUDA copy endpoint belongs to this runtime"};
    }
    // Storage bounds check.
    if (destination_offset_bytes > destination.size_bytes() ||
        byte_count > destination.size_bytes() - destination_offset_bytes || source_offset_bytes > source.size_bytes() ||
        byte_count > source.size_bytes() - source_offset_bytes)
    {
        throw std::out_of_range{"buffer copy exceeds source or destination storage"};
    }
    if (byte_count == 0 || (&destination == &source && destination_offset_bytes == source_offset_bytes))
    {
        return;
    }
    if (&destination == &source)
    {
        // Check for overlap.
        const auto distance = destination_offset_bytes > source_offset_bytes
                                  ? destination_offset_bytes - source_offset_bytes
                                  : source_offset_bytes - destination_offset_bytes;
        if (distance < byte_count)
        {
            throw std::invalid_argument{"buffer copy ranges overlap"};
        }
    }

    auto *target = (target_device ? target_device->data() : target_host->data()) + destination_offset_bytes;
    const auto *origin = (source_device ? source_device->data() : source_host->data()) + source_offset_bytes;
    DeviceGuard guard(device());
    if (target_device && source_device && target_device->device() != source_device->device())
    {
        // Copy data across CUDA devices.
        cuda_check(cudaMemcpyPeerAsync(target, static_cast<int>(target_device->device().index()), origin,
                                       static_cast<int>(source_device->device().index()), byte_count, nullptr),
                   "copy between CUDA devices");
    }
    else
    {
        // Select the cudaMemcpyKind based on the location of the source and target.
        const auto kind = !source_device   ? cudaMemcpyHostToDevice
                          : !target_device ? cudaMemcpyDeviceToHost
                                           : cudaMemcpyDeviceToDevice;
        cuda_check(cudaMemcpyAsync(target, origin, byte_count, kind, nullptr), "copy buffer data");
    }
    cuda_check(cudaStreamSynchronize(nullptr), "complete buffer copy");
    guard.restore();
}
} // namespace minitensor::detail::cuda
