#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/reduction/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_sum(
            DeviceRuntime &,
            const Primitive &primitive,
            std::span<const TensorView> inputs,
            MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &sum = dynamic_cast<const SumPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    reduce_values<T>(
                        inputs.front(),
                        output,
                        sum.axes(),
                        T{},
                        [](T accumulator, T value)
                        {
                            return accumulator + value;
                        });
                });
        }
    }

    void register_reduction_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(
            KernelKey{typeid(SumPrimitive), DeviceType::Cpu},
            run_sum);
    }
}
