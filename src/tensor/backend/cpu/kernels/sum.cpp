#include "registrations.hpp"

#include <cassert>
#include <array>
#include <span>
#include <cstddef>
#include <utility>
#include <typeinfo>

#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/ops/sum.hpp"
#include "tensor/storage/layout.hpp"
#include "../cpu_buffer.hpp"
#include "reduction.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_sum(DeviceRuntime&, const Primitive& primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            assert(output.dtype() == DType::Float32);

            const auto sum_primitive = dynamic_cast<const SumPrimitive*>(&primitive);
            assert(sum_primitive != nullptr);

            reduction_float32(inputs.front(), output, sum_primitive->axes(), 0.0F, [](float acc, float val) { return acc + val; });
        }
    }

    void register_sum(KernelRegistry& registry)
    {
        KernelKey key{ typeid(SumPrimitive), DeviceType::Cpu, DType::Float32 };
        registry.register_kernel(std::move(key), run_sum);
    }
}
