#include "registrations.hpp"

#include <span>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <cassert>

#include <minitensor/types.hpp>

#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/ops/full.hpp"
#include "../cpu_buffer.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_full(DeviceRuntime &device_runtime, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            (void)device_runtime;

            assert(inputs.empty());
            assert(output.dtype() == DType::Float32);
            assert(output.layout().is_contiguous(output.shape()));

            const auto &full = dynamic_cast<const FullPrimitive &>(primitive);
            const std::size_t numel = output.shape().numel();
            if (numel == 0)
            {
                return;
            }
            auto &buffer = dynamic_cast<CpuBuffer &>(output.buffer());
            auto *data = reinterpret_cast<float *>(buffer.data());
            const auto offset = static_cast<std::size_t>(output.layout().offset());
            std::fill_n(data + offset, numel, full.fill_value());
        }
    }

    void register_full(KernelRegistry &registry)
    {
        KernelKey key{typeid(FullPrimitive), DeviceType::Cpu, DType::Float32};
        registry.register_kernel(std::move(key), run_full);
    }
}