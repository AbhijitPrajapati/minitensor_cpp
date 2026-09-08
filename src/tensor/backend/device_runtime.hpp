#pragma once

#include "tensor/storage/buffer.hpp"

#include <cstddef>
#include <span>

namespace minitensor::detail
{
    class DeviceRuntime
    {
    public:
        DeviceRuntime(const DeviceRuntime &) = delete;
        DeviceRuntime &operator=(const DeviceRuntime &) = delete;
        virtual ~DeviceRuntime() = default;

        [[nodiscard]] virtual Device device() const noexcept = 0;
        [[nodiscard]] virtual BufferRef allocate(std::size_t size_bytes) = 0;
        virtual void copy_from_host(Buffer &destination, std::size_t destination_offset_bytes, std::span<const std::byte> source) = 0;
        virtual void copy_to_host(std::span<std::byte> destination, const Buffer &source, std::size_t source_offset_bytes) = 0;

    protected:
        DeviceRuntime() = default;
    };
}