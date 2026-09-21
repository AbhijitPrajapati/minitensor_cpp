#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/copy/copy.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_reshape(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            (void)dynamic_cast<const ReshapePrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    copy_to_dense<T>(inputs.front(), output);
                });
        }
    }

    void register_copy_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(ReshapePrimitive), DeviceType::Cpu, DType::Float32}, run_reshape);
    }
}
