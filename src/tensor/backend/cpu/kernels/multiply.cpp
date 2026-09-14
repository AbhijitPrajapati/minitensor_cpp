#include "registrations.hpp"

#include <span>
#include <cassert>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/ops/multiply.hpp"
#include "binary_elementwise.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_multiply(DeviceRuntime&, const Primitive& primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 2);

            (void)dynamic_cast<const MultiplyPrimitive&>(primitive);

            const TensorView& lhs = inputs[0];
            const TensorView& rhs = inputs[1];

            binary_elementwise_float32(lhs, rhs, output, [](float lhs_val, float rhs_val)
                                       {
                                           return lhs_val * rhs_val;
 });
        }
    }

    void register_multiply(KernelRegistry& registry)
    {
        KernelKey key{ typeid(MultiplyPrimitive), DeviceType::Cpu, DType::Float32 };
        registry.register_kernel(std::move(key), run_multiply);
    }
}
