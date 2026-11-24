#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/cuda/iteration/reduction.hpp"
#include "tensor/backend/cuda/iteration/strided.cuh"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor::detail::cuda
{
namespace
{
constexpr unsigned int reduction_threads = 256;

struct ReductionGeometry
{
    const std::int64_t *outer_extents{nullptr};
    const std::int64_t *outer_strides{nullptr};
    const std::int64_t *reduction_extents{nullptr};
    const std::int64_t *reduction_strides{nullptr};
    std::size_t outer_rank;
    std::size_t reduction_rank;
    std::size_t reduction_size;
    std::int64_t input_offset;
};

template <typename T, typename Operation>
__device__ T block_reduce(T value, T *scratch, Operation operation)
{
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int distance = blockDim.x / 2; distance != 0; distance /= 2)
    {
        if (threadIdx.x < distance)
        {
            scratch[threadIdx.x] = operation(scratch[threadIdx.x], scratch[threadIdx.x + distance]);
        }
        __syncthreads();
    }
    const T result = scratch[0];
    __syncthreads();
    return result;
}

__device__ std::int64_t reduction_offset(std::size_t index, const ReductionGeometry &geometry, std::int64_t base)
{
    std::int64_t offsets[1]{base};
    strided_offsets(index, geometry.reduction_extents, geometry.reduction_strides, geometry.reduction_rank, offsets);
    return offsets[0];
}

template <TensorElement T, typename Transform, typename Combine>
__device__ T reduce_values(const T *input, const ReductionGeometry &geometry, std::int64_t base, T *scratch, T initial,
                           Transform transform, Combine combine)
{
    T partial = initial;
    for (std::size_t index = threadIdx.x; index < geometry.reduction_size; index += blockDim.x)
    {
        partial = combine(partial, transform(input[reduction_offset(index, geometry, base)]));
    }
    return block_reduce(partial, scratch, combine);
}

template <TensorElement T, typename Operation>
__global__ void reduction_kernel(const T *input, T *output, std::size_t output_size, ReductionGeometry geometry,
                                 T empty_value, Operation operation)
{
    __shared__ T scratch[reduction_threads];
    for (std::size_t linear = blockIdx.x; linear < output_size; linear += gridDim.x)
    {
        if (geometry.reduction_size == 0)
        {
            if (threadIdx.x == 0)
                output[linear] = empty_value;
            continue;
        }
        std::int64_t outer[1]{geometry.input_offset};
        strided_offsets(linear, geometry.outer_extents, geometry.outer_strides, geometry.outer_rank, outer);
        // The operation is collective: all threads must participate in the same block reductions, including any
        // branches between multiple passes.
        const T result = operation(input, geometry, outer[0], scratch);
        if (threadIdx.x == 0)
            output[linear] = result;
    }
}

template <typename PrimitiveType, typename Operation>
void run_reduction(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output, float empty_value, Operation operation)
{
    assert(inputs.size() == 1);
    const auto &reduction = dynamic_cast<const PrimitiveType &>(primitive);
    const TensorView &input = inputs[0];
    assert(runtime.device() == output.device() && input.device() == output.device());
    assert(input.dtype() == output.dtype());
    assert(output.layout().is_contiguous(output.shape()));
    if (output.shape().numel() == 0)
        return;

    const ReductionPlan plan{input.shape(), input.layout(), reduction.axes()};
    assert(plan.output_size() == output.shape().numel());
    DeviceGuard guard(output.device());
    ReductionGeometry geometry{nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               plan.outer_shape().rank(),
                               plan.reduction_shape().rank(),
                               plan.reduction_size(),
                               plan.outer_layout().offset()};
    BufferRef metadata_buffer;
    if (plan.reduction_size() != 0 && input.shape().rank() != 0)
    {
        std::vector<std::int64_t> metadata;
        metadata.reserve(2 * input.shape().rank());
        const auto append = [&](const Shape &shape, const Layout &layout) {
            metadata.insert(metadata.end(), shape.begin(), shape.end());
            const auto strides = layout.strides();
            metadata.insert(metadata.end(), strides.begin(), strides.end());
        };
        append(plan.outer_shape(), plan.outer_layout());
        append(plan.reduction_shape(), plan.reduction_layout());
        const auto bytes = std::as_bytes(std::span{metadata});
        metadata_buffer = runtime.allocate(bytes.size());
        runtime.copy_from_host(*metadata_buffer, 0, bytes);
        geometry.outer_extents =
            reinterpret_cast<const std::int64_t *>(dynamic_cast<const CudaBuffer &>(*metadata_buffer).data());
        geometry.outer_strides = geometry.outer_extents + geometry.outer_rank;
        geometry.reduction_extents = geometry.outer_strides + geometry.outer_rank;
        geometry.reduction_strides = geometry.reduction_extents + geometry.reduction_rank;
    }
    const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(plan.output_size(), 65535));
    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        const T *source = plan.reduction_size() == 0 ? nullptr : data<T>(input);
        reduction_kernel<T><<<blocks, reduction_threads>>>(source, data<T>(output) + output.layout().offset(),
                                                           plan.output_size(), geometry, static_cast<T>(empty_value),
                                                           operation);
        cuda_check(cudaGetLastError(), "launch reduction kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete reduction kernel");
    metadata_buffer.reset();
    guard.restore();
}

void run_sum(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
             MutableTensorView output)
{
    run_reduction<SumPrimitive>(
        runtime, primitive, inputs, output, 0.0F,
        [] __device__(const auto *input, const ReductionGeometry &geometry, std::int64_t base, auto *scratch) {
            using T = std::remove_pointer_t<decltype(scratch)>;
            return reduce_values(
                input, geometry, base, scratch, T{0}, [] __device__(T value) { return value; },
                [] __device__(T lhs, T rhs) { return lhs + rhs; });
        });
}

void run_logsumexp(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output)
{
    run_reduction<LogSumExpPrimitive>(
        runtime, primitive, inputs, output, -std::numeric_limits<float>::infinity(),
        [] __device__(const auto *input, const ReductionGeometry &geometry, std::int64_t base, auto *scratch) {
            using T = std::remove_pointer_t<decltype(scratch)>;
            const auto maximum = [] __device__(T lhs, T rhs) {
                if (::isnan(lhs))
                    return lhs;
                if (::isnan(rhs))
                    return rhs;
                return lhs < rhs ? rhs : lhs;
            };
            const T max_value = reduce_values(
                input, geometry, base, scratch, -static_cast<T>(CUDART_INF_F), [] __device__(T value) { return value; },
                maximum);
            // The block-wide maximum is identical in every thread, making this
            // early return uniform and safe for the following collective pass.
            if (::isnan(max_value) || ::isinf(max_value))
                return max_value;
            const T exp_sum = reduce_values(
                input, geometry, base, scratch, T{0},
                [max_value] __device__(T value) { return ::exp(value - max_value); },
                [] __device__(T lhs, T rhs) { return lhs + rhs; });
            return max_value + ::log(exp_sum);
        });
}
} // namespace

void register_reduction_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{SumPrimitive::ID, DeviceType::Cuda}, run_sum);
    registry.register_kernel(KernelKey{LogSumExpPrimitive::ID, DeviceType::Cuda}, run_logsumexp);
}
} // namespace minitensor::detail::cuda
