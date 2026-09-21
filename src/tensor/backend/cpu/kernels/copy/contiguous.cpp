#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/copy/contiguous.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_contiguous(
            DeviceRuntime &,
            const Primitive &primitive,
            std::span<const TensorView> inputs,
            MutableTensorView output)
        {
            assert(inputs.size() == 1);
            assert(
                dynamic_cast<const ContiguousPrimitive *>(&primitive) != nullptr ||
                dynamic_cast<const ReshapePrimitive *>(&primitive) != nullptr);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    copy_to_contiguous<T>(inputs.front(), output);
                });
        }
    }

    void register_copy_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(
            KernelKey{typeid(ContiguousPrimitive), DeviceType::Cpu},
            run_contiguous);
        registry.register_kernel(
            KernelKey{typeid(ReshapePrimitive), DeviceType::Cpu},
            run_contiguous);
    }
}
