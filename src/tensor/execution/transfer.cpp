#include "transfer.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include <minitensor/types.hpp>

#include "environment.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_primitive.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/leaf.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/dense_size.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
namespace
{
ValueRef materialize_contiguous(const ValueRef &source)
{
    const auto *materialization = source->materialization();
    if (materialization != nullptr && materialization->layout().is_contiguous(source->spec().shape))
    {
        return source;
    }
    // ContiguousPrimitive shares already-contiguous evaluated storage, and
    // otherwise packs on the source device through its registered kernel.
    auto packed = apply_primitive(std::make_unique<ContiguousPrimitive>(), std::span<const ValueRef>{&source, 1});
    environment().evaluate(std::span<const ValueRef>{&packed, 1});
    return packed;
}

std::size_t byte_offset(const Materialization &materialization, DType dtype)
{
    const auto offset = materialization.layout().offset();
    const auto element_size = dtype_size(dtype);
    if (offset < 0)
    {
        throw std::logic_error{"contiguous value has negative storage offset"};
    }
    if (!std::in_range<std::size_t>(offset) ||
        static_cast<std::size_t>(offset) > std::numeric_limits<std::size_t>::max() / element_size)
    {
        throw std::overflow_error{"storage byte offset exceeds size_t"};
    }
    return static_cast<std::size_t>(offset) * element_size;
}
} // namespace

ValueRef make_value_from_host(TensorSpec spec, std::span<const std::byte> source)
{
    const std::size_t expected_size = dense_size_bytes(spec);
    if (source.size() != expected_size)
    {
        throw std::invalid_argument{"host data size does not match value specification"};
    }

    DeviceRuntime &runtime = environment().runtime_for(spec.device);
    BufferRef buffer = runtime.allocate(expected_size);
    if (!buffer)
    {
        throw std::runtime_error{"null buffer allocated"};
    }

    runtime.copy_from_host(*buffer, 0, source);
    Materialization materialization(std::move(buffer), Layout::contiguous(spec.shape));
    return make_materialized_leaf(std::move(spec), std::move(materialization));
}

void copy_value_to_host(std::span<std::byte> destination, const ValueRef &value)
{
    const TensorSpec &spec = value->spec();
    if (destination.size() != dense_size_bytes(spec))
    {
        throw std::invalid_argument{"host destination size does not match value specification"};
    }

    const auto packed = materialize_contiguous(value);
    if (destination.empty())
    {
        return;
    }

    const auto &materialization = *packed->materialization();
    DeviceRuntime &runtime = environment().runtime_for(spec.device);
    runtime.copy_to_host(destination, *materialization.buffer_ref(), byte_offset(materialization, spec.dtype));
}

ValueRef transfer_value(const ValueRef &source, Device destination)
{
    if (source->spec().device == destination)
    {
        return source;
    }
    TensorSpec spec = source->spec();
    spec.device = destination;
    const auto size = dense_size_bytes(spec);
    DeviceRuntime &destination_runtime = environment().runtime_for(destination);
    const auto packed = materialize_contiguous(source);
    BufferRef buffer = destination_runtime.allocate(size);
    if (!buffer)
    {
        throw std::runtime_error{"null buffer allocated"};
    }
    if (size != 0)
    {
        const auto &materialization = *packed->materialization();
        // Use the CUDA runtime only if CUDA in the destination or the source.
        DeviceRuntime *copy_runtime = &destination_runtime;
        if (destination.type() == DeviceType::Cpu && source->spec().device.type() == DeviceType::Cuda)
        {
            copy_runtime = &environment().runtime_for(source->spec().device);
        }
        copy_runtime->copy_buffer(*buffer, 0, *materialization.buffer_ref(), byte_offset(materialization, spec.dtype),
                                  size);
    }
    Materialization materialization{std::move(buffer), Layout::contiguous(spec.shape)};
    return make_materialized_leaf(std::move(spec), std::move(materialization));
}
} // namespace minitensor::detail
