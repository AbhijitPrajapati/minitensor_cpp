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

    private:
        Device device_;
    };
}