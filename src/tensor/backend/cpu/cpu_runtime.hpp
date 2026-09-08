#pragma once

#include <minitensor/types.hpp>

#include "tensor/backend/device_runtime.hpp"

namespace minitensor::detail::cpu
{
    class CpuRuntime final : public DeviceRuntime
    {
    public:
        explicit CpuRuntime(Device device = Device::cpu());
        [[nodiscard]] Device device() const noexcept override;
        [[nodiscard]] BufferRef allocate(std::size_t size_bytes) override;
        void copy_from_host(Buffer &destination, std::size_t destination_offset_bytes, std::span<const std::byte> source);
        void copy_to_host(std::span<std::byte> destination, const Buffer &source, std::size_t source_offset_bytes);

    private:
        Device device_;
    };
}