#pragma once

#include "tensor/storage/buffer.hpp"

#include <cstddef>

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

    protected:
        DeviceRuntime() = default;
    };
}