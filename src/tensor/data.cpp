#include <minitensor/data.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>

#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/execution/environment.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/tensor_access.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/iteration/elementwise.hpp"
#include "tensor/graph/ids.hpp"

namespace minitensor
{
    Tensor from_data(std::span<const float> data, Shape shape, TensorOptions options)
    {
        if (data.size() != shape.numel())
        {
            throw std::invalid_argument{"data size does not match requested shape"};
        }

        detail::TensorSpec spec{std::move(shape), DType::Float32, options.device};
        detail::DeviceRuntime &runtime = detail::environment().runtime_for(spec.device);
        detail::BufferRef buffer = runtime.allocate(detail::dense_size_bytes(spec));
        if (!buffer)
        {
            throw std::runtime_error{"null buffer allocated"};
        }
        runtime.copy_from_host(*buffer, 0, std::as_bytes(data));
        detail::Materialization materialization(std::move(buffer), detail::Layout::contiguous(spec.shape));
        auto value = std::make_shared<detail::Value>(detail::next_value_id(), std::move(spec));
        value->materialize(std::move(materialization));
        return detail::TensorAccess::make(std::move(value));
    }

    std::vector<float> to_vector(const Tensor &tensor)
    {
        eval(tensor);

        const detail::ValueRef &value = detail::TensorAccess::value(tensor);
        const detail::Materialization *materialization = value->materialization();
        if (!materialization)
        {
            throw std::logic_error{"value is unmaterialized after evaluation"};
        }

        const detail::TensorSpec &spec = value->spec();
        if (spec.dtype != DType::Float32)
        {
            throw std::logic_error{"unsupported datatype detected"};
        }

        std::vector<float> result(spec.shape.numel());
        if (result.empty())
        {
            return result;
        }

        detail::DeviceRuntime &runtime = detail::environment().runtime_for(spec.device);
        const detail::BufferRef &buffer = materialization->buffer_ref();
        const detail::Layout &layout = materialization->layout();

        // fast path if contiguous
        if (layout.is_contiguous(spec.shape))
        {
            assert(layout.offset() >= 0);
            const auto source_offset_bytes = static_cast<std::size_t>(layout.offset()) * sizeof(float);
            runtime.copy_to_host(std::as_writable_bytes(std::span<float>(result)), *buffer, source_offset_bytes);
            return result;
        }

        // general strided path

        std::vector<std::byte> host_storage(buffer->size_bytes());
        runtime.copy_to_host(host_storage, *buffer, 0);

        const std::array<detail::Layout, 1> layouts{materialization->layout()};
        detail::ElementwisePlan plan(tensor.shape(), layouts);
        plan.for_each(
            [&](Shape::size_type linear, std::span<const detail::Layout::offset_type> offsets)
            {
                assert(offsets.size() == 1);
                assert(offsets[0] >= 0);
                const auto source_offset_bytes = static_cast<std::size_t>(offsets[0]) * sizeof(float);
                assert(source_offset_bytes <= host_storage.size());
                std::memcpy(&result[linear], host_storage.data() + source_offset_bytes, sizeof(float));
            });

        return result;
    }

    float item(const Tensor &tensor)
    {
        if (tensor.numel() != 1)
        {
            throw std::invalid_argument{"item() requires a single tensor element"};
        }
        return to_vector(tensor).front();
    }
}
