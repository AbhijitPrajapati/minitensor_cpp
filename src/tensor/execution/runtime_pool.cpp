#include "runtime_pool.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/device_runtime.hpp"

#ifdef ENABLE_CUDA
#include "tensor/backend/cuda/cuda_runtime.hpp"
#endif

namespace minitensor::detail
{
RuntimePool::RuntimePool()
{
    register_runtime(std::make_unique<cpu::CpuRuntime>());
}

void RuntimePool::register_runtime(std::unique_ptr<DeviceRuntime> runtime)
{
    if (!runtime)
    {
        throw std::invalid_argument{"cannot register an empty runtime"};
    }
    const Device device = runtime->device();
    const auto result = runtimes_.try_emplace(device, std::move(runtime));
    if (!result.second)
    {
        throw std::logic_error{"a runtime is already registered for this device"};
    }
}

DeviceRuntime &RuntimePool::runtime_for(const Device &device) const
{
    const auto iterator = runtimes_.find(device);
    if (iterator != runtimes_.end())
    {
        return *iterator->second;
    }

    if (device.type() == DeviceType::Cuda)
    {
#ifdef ENABLE_CUDA
        const auto result = runtimes_.try_emplace(device, std::make_unique<cuda::CudaRuntime>(device));
        return *result.first->second;
#else
        throw std::runtime_error{"CUDA support is not enabled in this build"};
#endif
    }

    throw std::runtime_error{"no runtime is registered for this device"};
}
} // namespace minitensor::detail
