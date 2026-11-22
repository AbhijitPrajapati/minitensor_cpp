#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "tensor/backend/cuda/cuda_buffer.hpp"
#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/cuda/iteration/strided.cuh"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/gather.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{
namespace
{

template <TensorElement T, bool DestinationContiguous>
__global__ void copy_strided_kernel(const T *source, T *destination, std::size_t count, const std::int64_t *extents,
                                    const std::int64_t *strides, std::size_t rank, std::int64_t source_offset,
                                    std::int64_t destination_offset)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t linear = start; linear < count; linear += step)
    {
        std::int64_t offsets[DestinationContiguous ? 1 : 2]{source_offset};
        if constexpr (!DestinationContiguous)
        {
            offsets[1] = destination_offset;
        }
        strided_offsets(linear, extents, strides, rank, offsets);
        if constexpr (DestinationContiguous)
        {
            destination[destination_offset + static_cast<std::int64_t>(linear)] = source[offsets[0]];
        }
        else
        {
            destination[offsets[1]] = source[offsets[0]];
        }
    }
}

// Enqueue a copy between two affine layouts over the same logical shape while retaining scratch buffers until final
// synchronization.
template <TensorElement T>
void copy_strided(DeviceRuntime &runtime, const Shape &shape, const T *source, const Layout &source_layout,
                  T *destination, const Layout &destination_layout, std::vector<BufferRef> &metadata_buffers)
{
    assert(source_layout.rank() == shape.rank());
    assert(destination_layout.rank() == shape.rank());
    const std::size_t count = shape.numel();
    if (count == 0)
    {
        return;
    }

    // Fast path if both contiguous.
    const bool destination_contiguous = destination_layout.is_contiguous(shape);
    if (source_layout.is_contiguous(shape) && destination_contiguous)
    {
        cuda_check(cudaMemcpyAsync(destination + destination_layout.offset(), source + source_layout.offset(),
                                   count * sizeof(T), cudaMemcpyDeviceToDevice, nullptr),
                   "copy contiguous device data");
        return;
    }

    // Bundle extents and strides.
    const std::size_t rank = shape.rank();
    std::vector<std::int64_t> metadata;
    metadata.reserve(rank * (destination_contiguous ? 2 : 3));
    metadata.insert(metadata.end(), shape.begin(), shape.end());
    const auto source_strides = source_layout.strides();
    metadata.insert(metadata.end(), source_strides.begin(), source_strides.end());
    if (!destination_contiguous)
    {
        // Only add destination strides if the destination is not contiguous.
        const auto destination_strides = destination_layout.strides();
        metadata.insert(metadata.end(), destination_strides.begin(), destination_strides.end());
    }

    // Allocate the metadata buffer and obtain pointers.
    const auto metadata_bytes = std::as_bytes(std::span<const std::int64_t>(metadata));
    BufferRef metadata_buffer = runtime.allocate(metadata_bytes.size());
    runtime.copy_from_host(*metadata_buffer, 0, metadata_bytes);
    const auto &buffer = dynamic_cast<const CudaBuffer &>(*metadata_buffer);
    const auto *extents = reinterpret_cast<const std::int64_t *>(buffer.data());
    const auto *device_source_strides = extents + rank;
    metadata_buffers.push_back(std::move(metadata_buffer));

    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));

    // Split cases for contigous destinations.
    if (destination_contiguous)
    {
        copy_strided_kernel<T, true><<<blocks, threads>>>(source, destination, count, extents, device_source_strides,
                                                          rank, source_layout.offset(), destination_layout.offset());
    }
    else
    {
        copy_strided_kernel<T, false><<<blocks, threads>>>(source, destination, count, extents, device_source_strides,
                                                           rank, source_layout.offset(), destination_layout.offset());
    }
    cuda_check(cudaGetLastError(), "launch strided copy kernel");
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
    assert(input.shape().numel() == output.shape().numel());

    // Assert same shape for contiguous primitive.
    if (dynamic_cast<const ContiguousPrimitive *>(&primitive) != nullptr)
    {
        assert(input.shape() == output.shape());
    }
    else
    {
        (void)dynamic_cast<const ReshapePrimitive &>(primitive);
    }
    if (output.shape().numel() == 0)
    {
        return;
    }

    // Reshape changes the output shape, but preserves the input's logical order.
    const Layout destination_layout = Layout::contiguous(input.shape(), output.layout().offset());
    DeviceGuard guard(output.device());
    std::vector<BufferRef> metadata_buffers;
    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        copy_strided(runtime, input.shape(), data<T>(input), input.layout(), data<T>(output), destination_layout,
                     metadata_buffers);
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete contiguous copy");
    metadata_buffers.clear();
    guard.restore();
}

void run_concatenate(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                     MutableTensorView output)
{
    assert(!inputs.empty());
    assert(runtime.device() == output.device());
    assert(output.layout().is_contiguous(output.shape()));
    const auto &concatenate = dynamic_cast<const ConcatenatePrimitive &>(primitive);
    const auto axis = concatenate.axis();
    assert(axis < output.shape().rank());
    if (output.shape().numel() == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    std::vector<BufferRef> metadata_buffers;
    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output);
        Extent axis_start = 0;
        for (const TensorView &input : inputs)
        {
            assert(input.device() == runtime.device());
            assert(input.dtype() == output.dtype());
            assert(input.shape().rank() == output.shape().rank());
            if (input.shape().numel() != 0)
            {
                const auto output_strides = output.layout().strides();
                // Construct corresponding destination layout.
                const Layout destination_layout{
                    std::vector<Layout::stride_type>(output_strides.begin(), output_strides.end()),
                    output.layout().offset() + axis_start * output.layout().stride(axis)};
                copy_strided(runtime, input.shape(), data<T>(input), input.layout(), destination, destination_layout,
                             metadata_buffers);
            }
            axis_start += input.shape()[axis];
        }
        assert(axis_start == output.shape()[axis]);
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete concatenate copy");
    metadata_buffers.clear();
    guard.restore();
}

void run_basic_index_scatter(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                             MutableTensorView output)
{
    assert(inputs.size() == 1);
    assert(runtime.device() == output.device());
    assert(output.layout().is_contiguous(output.shape()));
    const auto &scatter = dynamic_cast<const BasicIndexScatterPrimitive &>(primitive);
    const auto &parameters = scatter.parameters();
    const TensorView &input = inputs[0];
    assert(input.device() == runtime.device());
    assert(input.dtype() == output.dtype());
    assert(input.shape() == parameters.output_shape());
    assert(output.shape() == parameters.input_shape());
    if (output.shape().numel() == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    std::vector<BufferRef> metadata_buffers;
    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output);
        cuda_check(
            cudaMemsetAsync(destination + output.layout().offset(), 0, output.shape().numel() * sizeof(T), nullptr),
            "zero basic index scatter output");
        if (input.shape().numel() != 0)
        {
            copy_strided(runtime, input.shape(), data<T>(input), input.layout(), destination,
                         parameters.derive_layout(output.layout()), metadata_buffers);
        }
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete basic index scatter copy");
    metadata_buffers.clear();
    guard.restore();
}

struct IndexingMetadata
{
    BufferRef buffer;
    const std::int64_t *extents;
    const std::int64_t *strides;
    int *error;
    std::size_t error_offset;
};

// Pack the iteration shape, each operand's strides, and a zero-initialized error
// flag into one allocation. The owner must survive the operation's completion.
IndexingMetadata make_indexing_metadata(DeviceRuntime &runtime, const Shape &shape, std::span<const Layout> layouts)
{
    std::vector<std::int64_t> metadata(shape.begin(), shape.end());
    for (const Layout &layout : layouts)
    {
        assert(layout.rank() == shape.rank());
        const auto strides = layout.strides();
        metadata.insert(metadata.end(), strides.begin(), strides.end());
    }
    const auto error_offset = metadata.size() * sizeof(std::int64_t);
    metadata.push_back(0);
    const auto bytes = std::as_bytes(std::span{metadata});
    BufferRef buffer = runtime.allocate(bytes.size());
    runtime.copy_from_host(*buffer, 0, bytes);
    auto *base = dynamic_cast<CudaBuffer &>(*buffer).data();
    const auto *extents = reinterpret_cast<const std::int64_t *>(base);
    return {std::move(buffer), extents, extents + shape.rank(), reinterpret_cast<int *>(base + error_offset),
            error_offset};
}

void check_indexing_error(DeviceRuntime &runtime, const IndexingMetadata &metadata, const char *message)
{
    std::array<int, 1> error{};
    runtime.copy_to_host(std::as_writable_bytes(std::span{error}), *metadata.buffer, metadata.error_offset);
    if (error[0] != 0)
    {
        throw std::out_of_range{message};
    }
}

template <TensorElement T>
__global__ void gather_kernel(const T *source, const std::int64_t *indices, T *destination, std::size_t count,
                              const std::int64_t *extents, const std::int64_t *strides, std::size_t rank,
                              std::int64_t source_offset, std::int64_t indices_offset, std::int64_t axis_extent,
                              std::int64_t axis_stride, int *error)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t linear = start; linear < count; linear += step)
    {
        std::int64_t offsets[2]{source_offset, indices_offset};
        strided_offsets(linear, extents, strides, rank, offsets);
        const std::int64_t index = indices[offsets[1]];
        if (index < 0 || index >= axis_extent)
        {
            atomicExch(error, 1);
            continue;
        }
        destination[linear] = source[offsets[0] + index * axis_stride];
    }
}

void run_gather(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    assert(inputs.size() == 2);
    const auto &gather = dynamic_cast<const GatherPrimitive &>(primitive);
    const TensorView &source = inputs[0];
    const TensorView &indices = inputs[1];
    const auto axis = gather.axis();
    assert(runtime.device() == output.device());
    assert(source.device() == output.device() && indices.device() == output.device());
    assert(source.dtype() == output.dtype() && indices.dtype() == DType::Int64);
    assert(source.shape().rank() == indices.shape().rank());
    assert(axis < source.shape().rank());
    assert(indices.shape() == output.shape());
    assert(output.layout().is_contiguous(output.shape()));
    const auto count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    const auto source_strides = source.layout().strides();
    std::vector<Layout::stride_type> base_strides(source_strides.begin(), source_strides.end());
    base_strides[axis] = 0;
    const std::array<Layout, 2> layouts{Layout{std::move(base_strides), source.layout().offset()}, indices.layout()};
    auto metadata = make_indexing_metadata(runtime, output.shape(), layouts);
    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));
    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        gather_kernel<T><<<blocks, threads>>>(
            data<T>(source), data<std::int64_t>(indices), data<T>(output) + output.layout().offset(), count,
            metadata.extents, metadata.strides, output.shape().rank(), source.layout().offset(),
            indices.layout().offset(), source.shape()[axis], source.layout().stride(axis), metadata.error);
        cuda_check(cudaGetLastError(), "launch gather kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete gather copy");
    check_indexing_error(runtime, metadata, "gather index is out of range");
    metadata.buffer.reset();
    guard.restore();
}

template <TensorElement T>
__global__ void gather_scatter_add_kernel(const T *updates, const std::int64_t *indices, T *destination,
                                          std::size_t count, const std::int64_t *extents, const std::int64_t *strides,
                                          std::size_t rank, std::int64_t updates_offset, std::int64_t indices_offset,
                                          std::int64_t destination_offset, std::int64_t axis_extent,
                                          std::int64_t axis_stride, int *error)
{
    static_assert(std::is_floating_point_v<T>);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t linear = start; linear < count; linear += step)
    {
        std::int64_t offsets[3]{updates_offset, indices_offset, destination_offset};
        strided_offsets(linear, extents, strides, rank, offsets);
        const std::int64_t index = indices[offsets[1]];
        if (index < 0 || index >= axis_extent)
        {
            atomicExch(error, 1);
            continue;
        }
        atomicAdd(destination + offsets[2] + index * axis_stride, updates[offsets[0]]);
    }
}

void run_gather_scatter_add(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                            MutableTensorView output)
{
    assert(inputs.size() == 2);
    const auto &scatter = dynamic_cast<const GatherScatterAddPrimitive &>(primitive);
    const TensorView &updates = inputs[0];
    const TensorView &indices = inputs[1];
    const auto axis = scatter.axis();
    assert(runtime.device() == output.device());
    assert(updates.device() == output.device() && indices.device() == output.device());
    assert(updates.dtype() == output.dtype() && indices.dtype() == DType::Int64);
    assert(updates.shape() == indices.shape());
    assert(updates.shape().rank() == output.shape().rank());
    assert(output.shape() == scatter.target_shape());
    assert(axis < output.shape().rank());
    assert(output.layout().is_contiguous(output.shape()));
    const auto count = updates.shape().numel();
    if (count == 0 && output.shape().numel() == 0)
    {
        return;
    }

    DeviceGuard guard(output.device());
    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output);
        if (output.shape().numel() != 0)
        {
            cuda_check(
                cudaMemsetAsync(destination + output.layout().offset(), 0, output.shape().numel() * sizeof(T), nullptr),
                "zero gather scatter add output");
        }
        if (count == 0)
        {
            cuda_check(cudaStreamSynchronize(nullptr), "complete empty gather scatter add");
            return;
        }

        const auto output_strides = output.layout().strides();
        std::vector<Layout::stride_type> base_strides(output_strides.begin(), output_strides.end());
        base_strides[axis] = 0;
        const std::array<Layout, 3> layouts{updates.layout(), indices.layout(),
                                            Layout{std::move(base_strides), output.layout().offset()}};
        auto metadata = make_indexing_metadata(runtime, updates.shape(), layouts);
        constexpr unsigned int threads = 256;
        constexpr std::size_t max_blocks = 65535;
        const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));
        gather_scatter_add_kernel<T><<<blocks, threads>>>(
            data<T>(updates), data<std::int64_t>(indices), destination, count, metadata.extents, metadata.strides,
            updates.shape().rank(), updates.layout().offset(), indices.layout().offset(), output.layout().offset(),
            output.shape()[axis], output.layout().stride(axis), metadata.error);
        cuda_check(cudaGetLastError(), "launch gather scatter add kernel");
        cuda_check(cudaStreamSynchronize(nullptr), "complete gather scatter add");
        check_indexing_error(runtime, metadata, "gather scatter add index is out of range");
    });
    guard.restore();
}
} // namespace

void register_copy_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{ContiguousPrimitive::ID, DeviceType::Cuda}, run_contiguous);
    registry.register_kernel(KernelKey{ReshapePrimitive::ID, DeviceType::Cuda}, run_contiguous);
    registry.register_kernel(KernelKey{ConcatenatePrimitive::ID, DeviceType::Cuda}, run_concatenate);
    registry.register_kernel(KernelKey{BasicIndexScatterPrimitive::ID, DeviceType::Cuda}, run_basic_index_scatter);
    registry.register_kernel(KernelKey{GatherPrimitive::ID, DeviceType::Cuda}, run_gather);
    registry.register_kernel(KernelKey{GatherScatterAddPrimitive::ID, DeviceType::Cuda}, run_gather_scatter_add);
}
} // namespace minitensor::detail::cuda
