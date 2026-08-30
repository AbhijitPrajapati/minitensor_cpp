#include <stdexcept>

#include "cpu_runtime.hpp"

#include "cpu_buffer.hpp"

namespace minitensor::detail::cpu
{
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
}