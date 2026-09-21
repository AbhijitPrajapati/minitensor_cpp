#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <cmath>
#include <span>
#include <type_traits>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/elementwise/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/exponential.hpp"
#include "tensor/primitives/elementwise/hyperbolic_tangent.hpp"
#include "tensor/primitives/elementwise/logarithm.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/square_root.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"

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

        void run_exponential(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_unary<ExponentialPrimitive>(
                primitive,
                inputs,
                output,
                [](auto input)
                {
                    return std::exp(input);
                });
        }

        void run_logarithm(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_unary<LogarithmPrimitive>(
                primitive,
                inputs,
                output,
                [](auto input)
                {
                    return std::log(input);
                });
        }

        void run_square_root(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_unary<SquareRootPrimitive>(
                primitive,
                inputs,
                output,
                [](auto input)
                {
                    return std::sqrt(input);
                });
        }

        void run_hyperbolic_tangent(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            run_unary<HyperbolicTangentPrimitive>(
                primitive,
                inputs,
                output,
                [](auto input)
                {
                    return std::tanh(input);
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

    void register_elementwise_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(NegatePrimitive), DeviceType::Cpu}, run_negate);
        registry.register_kernel(KernelKey{typeid(ExponentialPrimitive), DeviceType::Cpu}, run_exponential);
        registry.register_kernel(KernelKey{typeid(LogarithmPrimitive), DeviceType::Cpu}, run_logarithm);
        registry.register_kernel(KernelKey{typeid(SquareRootPrimitive), DeviceType::Cpu}, run_square_root);
        registry.register_kernel(KernelKey{typeid(HyperbolicTangentPrimitive), DeviceType::Cpu}, run_hyperbolic_tangent);
        registry.register_kernel(KernelKey{typeid(AddPrimitive), DeviceType::Cpu}, run_add);
        registry.register_kernel(KernelKey{typeid(SubtractPrimitive), DeviceType::Cpu}, run_subtract);
        registry.register_kernel(KernelKey{typeid(MultiplyPrimitive), DeviceType::Cpu}, run_multiply);
        registry.register_kernel(KernelKey{typeid(DividePrimitive), DeviceType::Cpu}, run_divide);
    }
}
