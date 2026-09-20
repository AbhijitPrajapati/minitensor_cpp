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
#include "tensor/ops/add.hpp"
#include "tensor/ops/divide.hpp"
#include "tensor/ops/multiply.hpp"
#include "tensor/ops/subtract.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        template <typename PrimitiveType, typename Operation>
        void run_binary(const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output, Operation operation)
        {
            assert(inputs.size() == 2);
            (void)dynamic_cast<const PrimitiveType &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    binary_elementwise<T>(inputs[0], inputs[1], output, operation);
                });
        }

        void run_add(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_binary<AddPrimitive>(
                primitive,
                inputs,
                output,
                [](auto lhs, auto rhs)
                {
                    return lhs + rhs;
                });
        }

        void run_multiply(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_binary<MultiplyPrimitive>(
                primitive,
                inputs,
                output,
                [](auto lhs, auto rhs)
                {
                    return lhs * rhs;
                });
        }

        void run_subtract(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_binary<SubtractPrimitive>(
                primitive,
                inputs,
                output,
                [](auto lhs, auto rhs)
                {
                    return lhs - rhs;
                });
        }

        void run_divide(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_binary<DividePrimitive>(
                primitive,
                inputs,
                output,
                [](auto lhs, auto rhs)
                {
                    return lhs / rhs;
                });
        }
    }

    void register_binary_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(AddPrimitive), DeviceType::Cpu, DType::Float32}, run_add);
        registry.register_kernel(KernelKey{typeid(SubtractPrimitive), DeviceType::Cpu, DType::Float32}, run_subtract);
        registry.register_kernel(KernelKey{typeid(MultiplyPrimitive), DeviceType::Cpu, DType::Float32}, run_multiply);
        registry.register_kernel(KernelKey{typeid(DividePrimitive), DeviceType::Cpu, DType::Float32}, run_divide);
    }
}
