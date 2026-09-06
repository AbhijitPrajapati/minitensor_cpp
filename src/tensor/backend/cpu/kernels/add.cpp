#include "registrations.hpp"

#include <span>
#include <cassert>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/ops/add.hpp"
#include "../cpu_buffer.hpp"
#include "binary_elementwise.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_add(DeviceRuntime &device_runtime, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            (void)device_runtime;

            assert(inputs.size() == 2);

            (void)dynamic_cast<const AddPrimitive &>(primitive);

            const TensorView &lhs = inputs[0];
            const TensorView &rhs = inputs[1];

            binary_elementwise_float32(lhs, rhs, output, [](float lhs_val, float rhs_val)
                                       { return lhs_val + rhs_val; });
        }
    }

    void register_add(KernelRegistry &registry)
    {
        KernelKey key{typeid(AddPrimitive), DeviceType::Cpu, DType::Float32};
        registry.register_kernel(std::move(key), run_add);
    }
}