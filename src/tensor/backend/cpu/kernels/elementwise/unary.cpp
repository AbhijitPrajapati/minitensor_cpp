#include "registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/elementwise/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/ops/negate.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        template <typename PrimitiveType, typename Operation>
        void run_unary(const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output, Operation operation)
        {
            assert(inputs.size() == 1);
            (void)dynamic_cast<const PrimitiveType &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    unary_elementwise<T>(inputs[0], output, operation);
                });
        }

        void run_negate(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_unary<NegatePrimitive>(
                primitive,
                inputs,
                output,
                [](auto input)
                {
                    return -input;
                });
        }
    }

    void register_unary_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(NegatePrimitive), DeviceType::Cpu, DType::Float32}, run_negate);
    }
}
