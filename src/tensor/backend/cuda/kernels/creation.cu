#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <span>
#include <type_traits>

#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{
namespace
{
template <TensorElement T>
__global__ void full_kernel(T *output, std::size_t count, T value)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < count; index += step)
    {
        output[index] = value;
    }
}

void run_full(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
              MutableTensorView output)
{
    assert(inputs.empty());
    (void)inputs;
    assert(runtime.device() == output.device());
    (void)runtime;
    assert(output.layout().is_contiguous(output.shape()));

    const auto &full = dynamic_cast<const FullPrimitive &>(primitive);
    const std::size_t count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    // Bound the grid size; each thread can fill multiple elements.
    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));

    DeviceGuard guard(output.device());

    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output) + output.layout().offset();
        const T value = full.fill_value().get<T>();
        full_kernel<T><<<blocks, threads>>>(destination, count, value);
        cuda_check(cudaGetLastError(), "launch full kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete full kernel");
    guard.restore();
}
} // namespace

void register_creation_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{FullPrimitive::ID, DeviceType::Cuda}, run_full);
}
} // namespace minitensor::detail::cuda
