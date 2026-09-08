#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include <minitensor/types.hpp>

#include "tensor/core/hash.hpp"
#include "tensor/backend/device_runtime.hpp"

namespace minitensor::detail
{
    struct DeviceHash final
    {
        [[nodiscard]] std::size_t operator()(const Device &device) const noexcept;
    };

    class RuntimeRegistry final
    {
    public:
        void register_runtime(std::unique_ptr<DeviceRuntime> runtime);
        [[nodiscard]] bool contains(const Device &device) const;
        [[nodiscard]] DeviceRuntime &get(const Device &device) const;

    private:
        std::unordered_map<Device, std::unique_ptr<DeviceRuntime>, DeviceHash> runtimes_;
    };
}
