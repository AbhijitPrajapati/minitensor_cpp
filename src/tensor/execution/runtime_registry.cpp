#include "runtime_registry.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "tensor/backend/device_runtime.hpp"

namespace minitensor::detail
{

    std::size_t DeviceHash::operator()(const Device &device) const noexcept
    {
        std::size_t result = 0;
        combine_hash(result, device.type());
        combine_hash(result, device.index());
        return result;
    }

    void RuntimeRegistry::register_runtime(std::unique_ptr<DeviceRuntime> runtime)
    {
        if (!runtime)
        {
            throw std::invalid_argument{"cannot register an empty runtime"};
        }
        const Device device = runtime->device();
        auto [iterator, inserted] = runtimes_.try_emplace(device, std::move(runtime));
        if (!inserted)
        {
            throw std::logic_error{"a runtime is already registered for this device"};
        }
    }

    DeviceRuntime &RuntimeRegistry::get(const Device &device) const
    {
        const auto iterator = runtimes_.find(device);
        if (iterator == runtimes_.end())
        {
            throw std::runtime_error{"no runtime is registered for this device"};
        }
        return *(iterator->second);
    }

    bool RuntimeRegistry::contains(const Device &device) const
    {
        return runtimes_.contains(device);
    }
}