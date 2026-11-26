#include "cpu_runtime.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>

#include <minitensor/types.hpp>

#include "cpu_buffer.hpp"
#include "tensor/storage/buffer.hpp"

namespace minitensor::detail::cpu
{
// The CpuRuntime simply uses memcpy to manage memory.

CpuRuntime::CpuRuntime(Device device) : device_{device}
{
    if (device_.type() != DeviceType::Cpu)
    {
        throw std::invalid_argument{"CpuRuntime requires a CPU device"};
    }
}

Device CpuRuntime::device() const noexcept
{
    return device_;
}

BufferRef CpuRuntime::allocate(std::size_t size_bytes)
{
    return std::make_shared<CpuBuffer>(size_bytes, device_);
}

void CpuRuntime::copy_from_host(Buffer &destination, std::size_t destination_offset_bytes,
                                std::span<const std::byte> source)
{
    auto *buffer = dynamic_cast<CpuBuffer *>(&destination);
    if (!buffer)
    {
        throw std::invalid_argument{"CpuRuntime requires a CpuBuffer destination"};
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

    auto *destination_bytes = static_cast<std::byte *>(buffer->data());
    std::memcpy(destination_bytes + destination_offset_bytes, source.data(), source.size());
}

void CpuRuntime::copy_to_host(std::span<std::byte> destination, const Buffer &source, std::size_t source_offset_bytes)
{
    const auto *buffer = dynamic_cast<const CpuBuffer *>(&source);
    if (!buffer)
    {
        throw std::invalid_argument{"CpuRuntime requires a CpuBuffer source"};
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
    std::memcpy(destination.data(), buffer->data() + source_offset_bytes, destination.size());
}
void CpuRuntime::copy_buffer(Buffer &destination, std::size_t destination_offset_bytes, const Buffer &source,
                             std::size_t source_offset_bytes, std::size_t byte_count)
{
    // Verify both endpoints are CPU buffers.
    auto *target = dynamic_cast<CpuBuffer *>(&destination);
    const auto *origin = dynamic_cast<const CpuBuffer *>(&source);
    if (!target || !origin)
    {
        throw std::invalid_argument{"CpuRuntime requires CpuBuffer endpoints"};
    }
    if (target->device() != device() && origin->device() != device())
    {
        throw std::invalid_argument{"neither copy endpoint belongs to this runtime"};
    }
    // Storage bounds check.
    if (destination_offset_bytes > target->size_bytes() ||
        byte_count > target->size_bytes() - destination_offset_bytes || source_offset_bytes > origin->size_bytes() ||
        byte_count > origin->size_bytes() - source_offset_bytes)
    {
        throw std::out_of_range{"buffer copy exceeds source or destination storage"};
    }
    if (byte_count == 0 || (target == origin && destination_offset_bytes == source_offset_bytes))
    {
        return;
    }
    if (target == origin)
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
    std::memcpy(target->data() + destination_offset_bytes, origin->data() + source_offset_bytes, byte_count);
}
} // namespace minitensor::detail::cpu
