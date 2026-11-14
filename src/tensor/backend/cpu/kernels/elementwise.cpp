#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/elementwise.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
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
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
namespace
{

template <TensorElement T, typename Operation>
    requires std::invocable<Operation &, T> && std::convertible_to<std::invoke_result_t<Operation &, T>, T>
void unary_elementwise(const TensorView &input, MutableTensorView output, Operation &&operation)
{
    assert(input.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);

    const Shape &output_shape = output.shape();
    assert(input.shape() == output_shape);
    assert(output.layout().is_contiguous(output_shape));

    if (output_shape.numel() == 0)
    {
        return;
    }

    const T *input_data = data<T>(input);
    T *output_data = data<T>(output);
    const auto input_offset = static_cast<std::size_t>(input.layout().offset());
    const auto output_offset = static_cast<std::size_t>(output.layout().offset());
    const Shape::size_type numel = output_shape.numel();

    // Contiguous fast path
    if (input.layout().is_contiguous(output_shape))
    {
        for (Shape::size_type i = 0; i < numel; ++i)
        {
            output_data[output_offset + i] = static_cast<T>(std::invoke(operation, input_data[input_offset + i]));
        }
        return;
    }

    // Regular strided path
    const std::array<Layout, 1> layouts{input.layout()};
    const ElementwisePlan iteration(output_shape, layouts);
    iteration.for_each_run([&](Shape::size_type linear, std::span<const Layout::offset_type> offsets,
                               std::span<const Layout::stride_type> strides, Shape::size_type run_size) {
        Layout::offset_type input_offset = offsets[0];
        const Layout::stride_type input_stride = strides[0];
        T *run_output = output_data + output_offset + linear;

        for (Shape::size_type i = 0; i < run_size; ++i)
        {
            assert(input_offset >= 0);
            run_output[i] = static_cast<T>(std::invoke(operation, input_data[static_cast<std::size_t>(input_offset)]));
            input_offset += input_stride;
        }
    });
}

template <typename PrimitiveType, typename Operation>
void run_unary(const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output,
               Operation operation)
{
    assert(inputs.size() == 1);
    (void)dynamic_cast<const PrimitiveType &>(primitive);

    dispatch_floating_dtype(
        output.dtype(), [&]<typename T>(std::type_identity<T>) { unary_elementwise<T>(inputs[0], output, operation); });
}

template <TensorElement T, typename Operation>
    requires std::invocable<Operation &, T, T> && std::convertible_to<std::invoke_result_t<Operation &, T, T>, T>
void binary_elementwise(const TensorView &lhs, const TensorView &rhs, MutableTensorView output, Operation &&operation)
{
    assert(lhs.dtype() == dtype_of_v<T>);
    assert(rhs.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);

    const Shape &output_shape = output.shape();
    assert(output.layout().is_contiguous(output_shape));

    if (output_shape.numel() == 0)
    {
        return;
    }

    const T *lhs_data = data<T>(lhs);
    const T *rhs_data = data<T>(rhs);
    T *output_data = data<T>(output);
    const auto output_offset = static_cast<std::size_t>(output.layout().offset());
    const Shape::size_type numel = output_shape.numel();

    // No-broadcast both-contiguous fast path
    if (lhs.shape() == output_shape && rhs.shape() == output_shape && lhs.layout().is_contiguous(output_shape) &&
        rhs.layout().is_contiguous(output_shape))
    {
        const auto lhs_offset = static_cast<std::size_t>(lhs.layout().offset());
        const auto rhs_offset = static_cast<std::size_t>(rhs.layout().offset());
        for (Shape::size_type i = 0; i < numel; ++i)
        {
            output_data[output_offset + i] =
                static_cast<T>(std::invoke(operation, lhs_data[lhs_offset + i], rhs_data[rhs_offset + i]));
        }
        return;
    }

    // Lhs scalar rhs contiguous fast path
    if (lhs.shape().numel() == 1 && rhs.shape() == output_shape && rhs.layout().is_contiguous(output_shape))
    {
        const T lhs_value = lhs_data[static_cast<std::size_t>(lhs.layout().offset())];
        const auto rhs_offset = static_cast<std::size_t>(rhs.layout().offset());
        for (Shape::size_type i = 0; i < numel; ++i)
        {
            output_data[output_offset + i] =
                static_cast<T>(std::invoke(operation, lhs_value, rhs_data[rhs_offset + i]));
        }
        return;
    }

    // rhs scalar lhs contiguous fast path
    if (rhs.shape().numel() == 1 && lhs.shape() == output_shape && lhs.layout().is_contiguous(output_shape))
    {
        const auto lhs_offset = static_cast<std::size_t>(lhs.layout().offset());
        const T rhs_value = rhs_data[static_cast<std::size_t>(rhs.layout().offset())];
        for (Shape::size_type i = 0; i < numel; ++i)
        {
            output_data[output_offset + i] =
                static_cast<T>(std::invoke(operation, lhs_data[lhs_offset + i], rhs_value));
        }
        return;
    }

    // Regular strided path
    const std::array<Layout, 2> layouts{lhs.layout().broadcasted_to(lhs.shape(), output_shape),
                                        rhs.layout().broadcasted_to(rhs.shape(), output_shape)};

    const ElementwisePlan iteration(output_shape, layouts);
    iteration.for_each_run([&](Shape::size_type linear, std::span<const Layout::offset_type> offsets,
                               std::span<const Layout::stride_type> strides, Shape::size_type run_size) {
        Layout::offset_type lhs_offset = offsets[0];
        Layout::offset_type rhs_offset = offsets[1];
        const Layout::stride_type lhs_stride = strides[0];
        const Layout::stride_type rhs_stride = strides[1];
        T *run_output = output_data + output_offset + linear;
        for (Shape::size_type i = 0; i < run_size; ++i)
        {
            assert(lhs_offset >= 0);
            assert(rhs_offset >= 0);
            run_output[i] = static_cast<T>(std::invoke(operation, lhs_data[static_cast<std::size_t>(lhs_offset)],
                                                       rhs_data[static_cast<std::size_t>(rhs_offset)]));
            lhs_offset += lhs_stride;
            rhs_offset += rhs_stride;
        }
    });
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
