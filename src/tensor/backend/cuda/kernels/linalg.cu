#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <minitensor/types.hpp>

#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/cuda/iteration/matmul.hpp"
#include "tensor/backend/cuda/iteration/strided.cuh"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{

namespace
{
// Each thread computes one output element.
template <TensorElement T>
__global__ void matmul_kernel(const T *lhs, const T *rhs, T *output, std::size_t count, std::size_t rows,
                              std::size_t columns, std::size_t contraction, MatmulPlan::MatrixStrides lhs_strides,
                              MatmulPlan::MatrixStrides rhs_strides, const std::int64_t *batch_extents,
                              const std::int64_t *batch_strides, std::size_t batch_rank, std::int64_t lhs_offset,
                              std::int64_t rhs_offset)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t linear = start; linear < count; linear += step)
    {
        // Find the output location.
        const auto column = static_cast<std::int64_t>(linear % columns);
        const std::size_t matrix_row = linear / columns;
        const auto row = static_cast<std::int64_t>(matrix_row % rows);
        const std::size_t batch = matrix_row / rows;

        // Obtain strided offsets from plan.
        std::int64_t offsets[2]{lhs_offset, rhs_offset};
        strided_offsets(batch, batch_extents, batch_strides, batch_rank, offsets);
        const std::int64_t lhs_row = offsets[0] + row * lhs_strides.row;
        const std::int64_t rhs_column = offsets[1] + column * rhs_strides.column;

        // Accumulate along the contraction dimension.
        T accumulator{};
        for (std::size_t k = 0; k < contraction; ++k)
        {
            const auto index = static_cast<std::int64_t>(k);
            accumulator += lhs[lhs_row + index * lhs_strides.column] * rhs[rhs_column + index * rhs_strides.row];
        }
        output[linear] = accumulator;
    }
}

void run_matmul(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    assert(inputs.size() == 2);
    (void)dynamic_cast<const MatmulPrimitive &>(primitive);
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

    const MatmulPlan plan{lhs.shape(), lhs.layout(), rhs.shape(), rhs.layout(), output.shape(), output.layout()};
    DeviceGuard guard(output.device());
    const auto batch_rank = plan.batch_shape().rank();
    BufferRef metadata_buffer;
    const std::int64_t *batch_extents = nullptr;
    const std::int64_t *batch_strides = nullptr;
    if (plan.contraction_size() != 0 && batch_rank != 0)
    {
        // Construct metadata buffer.
        std::vector<std::int64_t> metadata;
        metadata.reserve(3 * batch_rank);
        metadata.insert(metadata.end(), plan.batch_shape().begin(), plan.batch_shape().end());
        const auto lhs_batch_strides = plan.lhs_batch_layout().strides();
        const auto rhs_batch_strides = plan.rhs_batch_layout().strides();
        metadata.insert(metadata.end(), lhs_batch_strides.begin(), lhs_batch_strides.end());
        metadata.insert(metadata.end(), rhs_batch_strides.begin(), rhs_batch_strides.end());
        const auto bytes = std::as_bytes(std::span{metadata});
        metadata_buffer = runtime.allocate(bytes.size());
        runtime.copy_from_host(*metadata_buffer, 0, bytes);
        batch_extents =
            reinterpret_cast<const std::int64_t *>(dynamic_cast<const CudaBuffer &>(*metadata_buffer).data());
        batch_strides = batch_extents + batch_rank;
    }

    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output) + plan.output_offset();
        // An empty contraction still has nonempty, zero-valued output.
        if (plan.contraction_size() == 0)
        {
            cuda_check(cudaMemsetAsync(destination, 0, count * sizeof(T), nullptr), "zero empty matmul contraction");
            return;
        }
        constexpr unsigned int threads = 256;
        constexpr std::size_t max_blocks = 65535;
        const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));
        matmul_kernel<T><<<blocks, threads>>>(data<T>(lhs), data<T>(rhs), destination, count, plan.row_count(),
                                              plan.column_count(), plan.contraction_size(), plan.lhs_strides(),
                                              plan.rhs_strides(), batch_extents, batch_strides, batch_rank,
                                              plan.lhs_batch_layout().offset(), plan.rhs_batch_layout().offset());
        cuda_check(cudaGetLastError(), "launch matmul kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete matmul kernel");
    metadata_buffer.reset();
    guard.restore();
}
} // namespace

void register_linalg_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{MatmulPrimitive::ID, DeviceType::Cuda}, run_matmul);
}
} // namespace minitensor::detail::cuda
