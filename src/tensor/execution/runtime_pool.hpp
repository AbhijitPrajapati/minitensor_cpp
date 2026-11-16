#pragma once

#include <memory>
#include <unordered_map>

#include <minitensor/types.hpp>

#include "device_hash.hpp"
#include "tensor/backend/device_runtime.hpp"

namespace minitensor::detail
{
// Owns one runtime per device: CPU zero is eager, while CUDA runtimes are cached on demand.
class RuntimePool final
{
  public:
    RuntimePool();

    [[nodiscard]] DeviceRuntime &runtime_for(const Device &device) const;
    void register_runtime(std::unique_ptr<DeviceRuntime> runtime);

  private:
    mutable std::unordered_map<Device, std::unique_ptr<DeviceRuntime>, DeviceHash> runtimes_;
};
} // namespace minitensor::detail
