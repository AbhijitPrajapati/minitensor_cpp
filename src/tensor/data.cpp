#include <minitensor/data.hpp>

#include <span>
#include <stdexcept>
#include <utility>
#include <cstddef>
#include <memory>
#include <vector>
#include <cassert>
#include <array>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>
#include <minitensor/evaluation.hpp>

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

namespace minitensor
{
    // Tensor from_data(std::span<const float> data, Shape shape, TensorOptions options)
    // {
    //     if (data.size() != shape.numel())
    //     {
    //         throw std::invalid_argument{"data size does not match requested shape"};
    //     }

    //     detail::TensorSpec spec{std::move(shape), DType::Float32, options.device};
    //     auto &runtime = detail::environment().runtime_for(spec.device);
    //     detail::BufferRef buffer = runtime.allocate(detail::dense_size_bytes(spec));
    //     runtime.copy_from_host(*buffer, 0, std::as_bytes(data));
    //     detail::Materialization materialization(std::move(buffer), detail::Layout::contiguous(spec.shape));
    //     auto value = std::make_shared<detail::Value>(spec);
    //     value->materialize(std::move(materialization));
    //     return detail::TensorAccess::make(std::move(value));
    // }

    // std::vector<float> to_vector(const Tensor &tensor)
    // {
    //     eval(tensor);

    //     detail::ValueRef value = detail::TensorAccess::value(tensor);
    //     auto &runtime = detail::environment().runtime_for(value->spec().device);

    //     auto materialization = value->materialization();
    //     if (!materialization)
    //     {
    //         throw std::logic_error{"value is unmaterialized after evaluation"};
    //     }

    //     std::vector<float> result(tensor.numel());
    //     if (result.empty())
    //     {
    //         return result;
    //     }

    //     if (tensor.is_contiguous())
    //     {
    //         const auto offset = materialization->layout().offset();
    //         assert(offset >= 0);
    //         const auto element_offset = static_cast<std::size_t>(offset);
    //         const auto byte_offset = element_offset * sizeof(float);
    //         runtime.copy_to_host(std::as_writable_bytes(std::span(result)), *materialization->buffer_ref(), byte_offset);
    //         return result;
    //     }

    //     std::vector<std::byte> host_storage(materialization->buffer_ref()->size_bytes());
    //     runtime.copy_to_host(host_storage, *materialization->buffer_ref(), 0);
    //     const std::array<detail::Layout, 1> layouts{materialization->layout()};
    //     detail::ElementwisePlan plan(tensor.shape(), layouts);
    //     plan.for_each(
    //         [&](Shape::size_type linear, std::span<const detail::Layout::offset_type> offsets)
    //         {
    //             assert(offsets.size() == 1);
    //             assert(offsets[0] >= 0);
    //             const auto source_idx = static_cast<std::size_t>(offsets[0]);
    //             std::memcpy(&result[linear], host_storage.data() + source_idx * sizeof(float), sizeof(float));
    //         });
    //     return result;
    // }

    // float item(const Tensor &tensor)
    // {
    //     if (tensor.numel() != 1)
    //     {
    //         throw std::invalid_argument{"item() requires a single tensor element"};
    //     }
    //     return to_vector(tensor).front();
    // }
}