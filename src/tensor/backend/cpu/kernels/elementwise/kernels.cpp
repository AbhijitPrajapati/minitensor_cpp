#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/elementwise/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/elementwise/absolute_value.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/exponential.hpp"
#include "tensor/primitives/elementwise/hyperbolic_tangent.hpp"
#include "tensor/primitives/elementwise/logaddexp.hpp"
#include "tensor/primitives/elementwise/logarithm.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/sigmoid.hpp"
#include "tensor/primitives/elementwise/sign.hpp"
#include "tensor/primitives/elementwise/square_root.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"

namespace minitensor::detail::cpu
{
namespace
{
template <typename PrimitiveType, typename Operation>
void run_unary(const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output,
               Operation operation)
{
    assert(inputs.size() == 1);
    (void)dynamic_cast<const PrimitiveType &>(primitive);

    dispatch_floating_dtype(
        output.dtype(), [&]<typename T>(std::type_identity<T>) { unary_elementwise<T>(inputs[0], output, operation); });
}

template <typename PrimitiveType, typename Operation>
void run_binary(const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output,
                Operation operation)
{
    assert(inputs.size() == 2);
    (void)dynamic_cast<const PrimitiveType &>(primitive);

    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        binary_elementwise<T>(inputs[0], inputs[1], output, operation);
    });
}

void run_negate(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    run_unary<NegatePrimitive>(primitive, inputs, output, [](auto input) { return -input; });
}

void run_exponential(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                     MutableTensorView output)
{
    run_unary<ExponentialPrimitive>(primitive, inputs, output, [](auto input) { return std::exp(input); });
}

void run_logarithm(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output)
{
    run_unary<LogarithmPrimitive>(primitive, inputs, output, [](auto input) { return std::log(input); });
}

void run_square_root(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                     MutableTensorView output)
{
    run_unary<SquareRootPrimitive>(primitive, inputs, output, [](auto input) { return std::sqrt(input); });
}

void run_hyperbolic_tangent(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                            MutableTensorView output)
{
    run_unary<HyperbolicTangentPrimitive>(primitive, inputs, output, [](auto input) { return std::tanh(input); });
}

void run_absolute_value(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                        MutableTensorView output)
{
    run_unary<AbsoluteValuePrimitive>(primitive, inputs, output, [](auto input) { return std::abs(input); });
}

void run_sign(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
{
    run_unary<SignPrimitive>(primitive, inputs, output, [](auto input) {
        using T = decltype(input);
        if (input > T{0})
        {
            return T{1};
        }
        if (input < T{0})
        {
            return T{-1};
        }
        return input;
    });
}

void run_sigmoid(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                 MutableTensorView output)
{
    run_unary<SigmoidPrimitive>(primitive, inputs, output, [](auto input) {
        using T = decltype(input);

        if (input >= T{0})
        {
            const T exponential = std::exp(-input);
            return T{1} / (T{1} + exponential);
        }

        const T exponential = std::exp(input);
        return exponential / (T{1} + exponential);
    });
}

void run_add(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
{
    run_binary<AddPrimitive>(primitive, inputs, output, [](auto lhs, auto rhs) { return lhs + rhs; });
}

void run_subtract(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                  MutableTensorView output)
{
    run_binary<SubtractPrimitive>(primitive, inputs, output, [](auto lhs, auto rhs) { return lhs - rhs; });
}

void run_multiply(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                  MutableTensorView output)
{
    run_binary<MultiplyPrimitive>(primitive, inputs, output, [](auto lhs, auto rhs) { return lhs * rhs; });
}

void run_divide(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    run_binary<DividePrimitive>(primitive, inputs, output, [](auto lhs, auto rhs) { return lhs / rhs; });
}

void run_logaddexp(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output)
{
    run_binary<LogAddExpPrimitive>(primitive, inputs, output, [](auto lhs, auto rhs) {
        if (std::isnan(lhs) || std::isnan(rhs))
        {
            return lhs + rhs;
        }

        const auto maximum = std::max(lhs, rhs);
        if (std::isinf(maximum))
        {
            return maximum;
        }

        return maximum + std::log1p(std::exp(-std::abs(lhs - rhs)));
    });
}
} // namespace

void register_elementwise_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{NegatePrimitive::ID, DeviceType::Cpu}, run_negate);
    registry.register_kernel(KernelKey{ExponentialPrimitive::ID, DeviceType::Cpu}, run_exponential);
    registry.register_kernel(KernelKey{LogarithmPrimitive::ID, DeviceType::Cpu}, run_logarithm);
    registry.register_kernel(KernelKey{SquareRootPrimitive::ID, DeviceType::Cpu}, run_square_root);
    registry.register_kernel(KernelKey{HyperbolicTangentPrimitive::ID, DeviceType::Cpu}, run_hyperbolic_tangent);
    registry.register_kernel(KernelKey{AbsoluteValuePrimitive::ID, DeviceType::Cpu}, run_absolute_value);
    registry.register_kernel(KernelKey{SignPrimitive::ID, DeviceType::Cpu}, run_sign);
    registry.register_kernel(KernelKey{SigmoidPrimitive::ID, DeviceType::Cpu}, run_sigmoid);
    registry.register_kernel(KernelKey{AddPrimitive::ID, DeviceType::Cpu}, run_add);
    registry.register_kernel(KernelKey{SubtractPrimitive::ID, DeviceType::Cpu}, run_subtract);
    registry.register_kernel(KernelKey{MultiplyPrimitive::ID, DeviceType::Cpu}, run_multiply);
    registry.register_kernel(KernelKey{DividePrimitive::ID, DeviceType::Cpu}, run_divide);
    registry.register_kernel(KernelKey{LogAddExpPrimitive::ID, DeviceType::Cpu}, run_logaddexp);
}
} // namespace minitensor::detail::cpu
