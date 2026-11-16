#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "tensor/backend/cuda/cuda_buffer.hpp"
#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{
namespace
{

template <TensorElement T>
__global__ void contiguous_kernel(const T *source, T *destination, std::size_t count, const std::int64_t *extents,
                                  const std::int64_t *strides, std::size_t rank, std::int64_t source_offset)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t linear = start; linear < count; linear += step)
    {
        std::size_t remaining = linear;
        std::int64_t offset = source_offset;

        for (std::size_t axis = rank; axis > 0;)
        {
            --axis;

            const auto extent = static_cast<std::size_t>(extents[axis]);
            const auto coordinate = static_cast<std::int64_t>(remaining % extent);

            remaining /= extent;
            offset += coordinate * strides[axis];
        }
        destination[linear] = source[offset];
    }
}

void run_contiguous(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                    MutableTensorView output)
{
    assert(inputs.size() == 1);
    assert(runtime.device() == output.device());
    assert(output.layout().is_contiguous(output.shape()));

    const TensorView &input = inputs[0];
    assert(runtime.device() == input.device());
    assert(input.dtype() == output.dtype());
    assert(input.shape() == output.shape());

    (void)dynamic_cast<const ContiguousPrimitive &>(primitive);

    const std::size_t count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    const bool input_contiguous = input.layout().is_contiguous(input.shape());
    const std::size_t rank = input.shape().rank();

    // Store extent and stride on device buffer for kernel arguments.
    BufferRef metadata_buffer;
    const std::int64_t *extents = nullptr;
    const std::int64_t *strides = nullptr;
    if (!input_contiguous)
    {
        // Collect metadata on the host.
        std::vector<std::int64_t> metadata(input.shape().begin(), input.shape().end());
        const auto input_strides = input.layout().strides();
        metadata.insert(metadata.end(), input_strides.begin(), input_strides.end());

        // Convert to bytes and copy to the device buffer.
        const auto metadata_bytes = std::as_bytes(std::span<const std::int64_t>(metadata));
        metadata_buffer = runtime.allocate(metadata_bytes.size());
        runtime.copy_from_host(*metadata_buffer, 0, metadata_bytes);

        // Navigate to the extent and stride positions.
        const auto &buffer = dynamic_cast<const CudaBuffer &>(*metadata_buffer);
        extents = reinterpret_cast<const std::int64_t *>(buffer.data());
        strides = extents + rank;
    }

    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        const T *source = data<T>(input);
        T *destination = data<T>(output) + output.layout().offset();

        if (input_contiguous)
        {
            // Contiguous fast path
            cuda_check(
                cudaMemcpy(destination, source + input.layout().offset(), count * sizeof(T), cudaMemcpyDeviceToDevice),
                "copy contiguous device data");
        }
        else
        {
            // Strided kernel
            constexpr unsigned int threads = 256;
            constexpr std::size_t max_blocks = 65535;
            const auto blocks =
                static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));
            contiguous_kernel<T>
                <<<blocks, threads>>>(source, destination, count, extents, strides, rank, input.layout().offset());
            cuda_check(cudaGetLastError(), "launch contiguous kernel");
        }
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete contiguous copy");
    metadata_buffer.reset();
    guard.restore();
}
} // namespace

void register_copy_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{ContiguousPrimitive::ID, DeviceType::Cuda}, run_contiguous);
}
} // namespace minitensor::detail::cuda
