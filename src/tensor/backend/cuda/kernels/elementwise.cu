#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <minitensor/types.hpp>

#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/cuda/iteration/strided.cuh"
#include "tensor/backend/device_runtime.hpp"
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

namespace minitensor::detail::cuda
{
namespace
{
template <TensorElement T, typename Operation>
__global__ void unary_linear_kernel(const T *input, T *output, std::size_t count, Operation operation)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < count; index += step)
    {
        output[index] = static_cast<T>(operation(input[index]));
    }
}

template <TensorElement T, typename Operation>
__global__ void unary_strided_kernel(const T *input, T *output, std::size_t count, const std::int64_t *extents,
                                     const std::int64_t *strides, std::size_t rank, std::int64_t input_offset,
                                     Operation operation)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < count; index += step)
    {
        std::int64_t offsets[1]{input_offset};
        strided_offsets(index, extents, strides, rank, offsets);
        output[index] = static_cast<T>(operation(input[offsets[0]]));
    }
}

// Each operand step is one for contiguous traversal or zero for a scalar.
template <TensorElement T, typename Operation>
__global__ void binary_linear_kernel(const T *lhs, const T *rhs, T *output, std::size_t count, std::size_t lhs_step,
                                     std::size_t rhs_step, Operation operation)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < count; index += step)
    {
        output[index] = static_cast<T>(operation(lhs[index * lhs_step], rhs[index * rhs_step]));
    }
}

template <TensorElement T, typename Operation>
__global__ void binary_strided_kernel(const T *lhs, const T *rhs, T *output, std::size_t count,
                                      const std::int64_t *extents, const std::int64_t *strides, std::size_t rank,
                                      std::int64_t lhs_offset, std::int64_t rhs_offset, Operation operation)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < count; index += step)
    {
        std::int64_t offsets[2]{lhs_offset, rhs_offset};
        strided_offsets(index, extents, strides, rank, offsets);
        output[index] = static_cast<T>(operation(lhs[offsets[0]], rhs[offsets[1]]));
    }
}
struct StridedMetadata
{
    BufferRef buffer;
    const std::int64_t *extents{nullptr};
    const std::int64_t *strides{nullptr};
};

// Pack the iteration shape and operand strides into one allocation. The owner must survive the operation's completion.
StridedMetadata make_strided_metadata(DeviceRuntime &runtime, const Shape &shape, std::span<const Layout> layouts)
{
    std::vector<std::int64_t> metadata;
    metadata.reserve(shape.rank() * (layouts.size() + 1));
    metadata.insert(metadata.end(), shape.begin(), shape.end());
    for (const Layout &layout : layouts)
    {
        assert(layout.rank() == shape.rank());
        const auto strides = layout.strides();
        metadata.insert(metadata.end(), strides.begin(), strides.end());
    }
    const auto bytes = std::as_bytes(std::span{metadata});
    BufferRef buffer = runtime.allocate(bytes.size());
    runtime.copy_from_host(*buffer, 0, bytes);
    const auto *extents = reinterpret_cast<const std::int64_t *>(dynamic_cast<const CudaBuffer &>(*buffer).data());
    return {std::move(buffer), extents, extents + shape.rank()};
}

template <typename PrimitiveType, typename Operation>
void run_unary(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
               MutableTensorView output, Operation operation)
{
    assert(inputs.size() == 1);
    (void)dynamic_cast<const PrimitiveType &>(primitive);
    const TensorView &input = inputs[0];
    assert(runtime.device() == output.device());
    assert(input.device() == output.device());
    assert(input.dtype() == output.dtype());
    assert(input.shape() == output.shape());
    assert(output.layout().is_contiguous(output.shape()));
    const auto count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    const bool linear = input.layout().is_contiguous(input.shape());
    StridedMetadata metadata;
    if (!linear)
    {
        // Construct strided metadata for non-linear inputs.
        const std::array<Layout, 1> layouts{input.layout()};
        metadata = make_strided_metadata(runtime, output.shape(), layouts);
    }
    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));
    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        const T *source = data<T>(input);
        T *destination = data<T>(output) + output.layout().offset();
        if (linear)
        {
            unary_linear_kernel<T>
                <<<blocks, threads>>>(source + input.layout().offset(), destination, count, operation);
        }
        else
        {
            unary_strided_kernel<T><<<blocks, threads>>>(source, destination, count, metadata.extents, metadata.strides,
                                                         output.shape().rank(), input.layout().offset(), operation);
        }
        cuda_check(cudaGetLastError(), "launch unary elementwise kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete unary elementwise kernel");
    metadata.buffer.reset();
    guard.restore();
}

template <typename PrimitiveType, typename Operation>
void run_binary(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output, Operation operation)
{
    assert(inputs.size() == 2);
    (void)dynamic_cast<const PrimitiveType &>(primitive);
    const TensorView &lhs = inputs[0];
    const TensorView &rhs = inputs[1];
    assert(runtime.device() == output.device());
    assert(lhs.device() == output.device() && rhs.device() == output.device());
    assert(lhs.dtype() == output.dtype() && rhs.dtype() == output.dtype());
    assert(output.layout().is_contiguous(output.shape()));
    const auto count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    const bool lhs_scalar = lhs.shape().numel() == 1;
    const bool rhs_scalar = rhs.shape().numel() == 1;
    const bool linear = (lhs_scalar || (lhs.shape() == output.shape() && lhs.layout().is_contiguous(output.shape()))) &&
                        (rhs_scalar || (rhs.shape() == output.shape() && rhs.layout().is_contiguous(output.shape())));
    StridedMetadata metadata;
    if (!linear)
    {
        // Construct strided metadata if either input is non-linear.
        const std::array<Layout, 2> layouts{lhs.layout().broadcasted_to(lhs.shape(), output.shape()),
                                            rhs.layout().broadcasted_to(rhs.shape(), output.shape())};
        metadata = make_strided_metadata(runtime, output.shape(), layouts);
    }
    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));
    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        const T *lhs_data = data<T>(lhs);
        const T *rhs_data = data<T>(rhs);
        T *destination = data<T>(output) + output.layout().offset();
        if (linear)
        {
            binary_linear_kernel<T><<<blocks, threads>>>(lhs_data + lhs.layout().offset(),
                                                         rhs_data + rhs.layout().offset(), destination, count,
                                                         lhs_scalar ? 0 : 1, rhs_scalar ? 0 : 1, operation);
        }
        else
        {
            binary_strided_kernel<T><<<blocks, threads>>>(lhs_data, rhs_data, destination, count, metadata.extents,
                                                          metadata.strides, output.shape().rank(),
                                                          lhs.layout().offset(), rhs.layout().offset(), operation);
        }
        cuda_check(cudaGetLastError(), "launch binary elementwise kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete binary elementwise kernel");
    metadata.buffer.reset();
    guard.restore();
}
void run_negate(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    run_unary<NegatePrimitive>(runtime, primitive, inputs, output, [] __device__(auto input) { return -input; });
}

void run_exponential(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                     MutableTensorView output)
{
    run_unary<ExponentialPrimitive>(runtime, primitive, inputs, output,
                                    [] __device__(auto input) { return ::exp(input); });
}

void run_logarithm(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output)
{
    run_unary<LogarithmPrimitive>(runtime, primitive, inputs, output,
                                  [] __device__(auto input) { return ::log(input); });
}

void run_square_root(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                     MutableTensorView output)
{
    run_unary<SquareRootPrimitive>(runtime, primitive, inputs, output,
                                   [] __device__(auto input) { return ::sqrt(input); });
}

void run_hyperbolic_tangent(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                            MutableTensorView output)
{
    run_unary<HyperbolicTangentPrimitive>(runtime, primitive, inputs, output,
                                          [] __device__(auto input) { return ::tanh(input); });
}

void run_absolute_value(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                        MutableTensorView output)
{
    run_unary<AbsoluteValuePrimitive>(runtime, primitive, inputs, output,
                                      [] __device__(auto input) { return ::fabs(input); });
}

void run_sign(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
              MutableTensorView output)
{
    run_unary<SignPrimitive>(runtime, primitive, inputs, output, [] __device__(auto input) {
        using T = decltype(input);
        if (input > T{0})
            return T{1};
        if (input < T{0})
            return T{-1};
        return input;
    });
}

void run_sigmoid(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                 MutableTensorView output)
{
    run_unary<SigmoidPrimitive>(runtime, primitive, inputs, output, [] __device__(auto input) {
        using T = decltype(input);
        if (input >= T{0})
        {
            const T exponential = ::exp(-input);
            return T{1} / (T{1} + exponential);
        }
        const T exponential = ::exp(input);
        return exponential / (T{1} + exponential);
    });
}

void run_add(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
             MutableTensorView output)
{
    run_binary<AddPrimitive>(runtime, primitive, inputs, output,
                             [] __device__(auto lhs, auto rhs) { return lhs + rhs; });
}

void run_subtract(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                  MutableTensorView output)
{
    run_binary<SubtractPrimitive>(runtime, primitive, inputs, output,
                                  [] __device__(auto lhs, auto rhs) { return lhs - rhs; });
}

void run_multiply(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                  MutableTensorView output)
{
    run_binary<MultiplyPrimitive>(runtime, primitive, inputs, output,
                                  [] __device__(auto lhs, auto rhs) { return lhs * rhs; });
}

void run_divide(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    run_binary<DividePrimitive>(runtime, primitive, inputs, output,
                                [] __device__(auto lhs, auto rhs) { return lhs / rhs; });
}

void run_logaddexp(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output)
{
    run_binary<LogAddExpPrimitive>(runtime, primitive, inputs, output, [] __device__(auto lhs, auto rhs) {
        if (::isnan(lhs) || ::isnan(rhs))
        {
            return lhs + rhs;
        }
        const auto maximum = lhs < rhs ? rhs : lhs;
        if (::isinf(maximum))
        {
            return maximum;
        }
        return maximum + ::log1p(::exp(-::fabs(lhs - rhs)));
    });
}
} // namespace

void register_elementwise_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{NegatePrimitive::ID, DeviceType::Cuda}, run_negate);
    registry.register_kernel(KernelKey{ExponentialPrimitive::ID, DeviceType::Cuda}, run_exponential);
    registry.register_kernel(KernelKey{LogarithmPrimitive::ID, DeviceType::Cuda}, run_logarithm);
    registry.register_kernel(KernelKey{SquareRootPrimitive::ID, DeviceType::Cuda}, run_square_root);
    registry.register_kernel(KernelKey{HyperbolicTangentPrimitive::ID, DeviceType::Cuda}, run_hyperbolic_tangent);
    registry.register_kernel(KernelKey{AbsoluteValuePrimitive::ID, DeviceType::Cuda}, run_absolute_value);
    registry.register_kernel(KernelKey{SignPrimitive::ID, DeviceType::Cuda}, run_sign);
    registry.register_kernel(KernelKey{SigmoidPrimitive::ID, DeviceType::Cuda}, run_sigmoid);
    registry.register_kernel(KernelKey{AddPrimitive::ID, DeviceType::Cuda}, run_add);
    registry.register_kernel(KernelKey{SubtractPrimitive::ID, DeviceType::Cuda}, run_subtract);
    registry.register_kernel(KernelKey{MultiplyPrimitive::ID, DeviceType::Cuda}, run_multiply);
    registry.register_kernel(KernelKey{DividePrimitive::ID, DeviceType::Cuda}, run_divide);
    registry.register_kernel(KernelKey{LogAddExpPrimitive::ID, DeviceType::Cuda}, run_logaddexp);
}
} // namespace minitensor::detail::cuda
