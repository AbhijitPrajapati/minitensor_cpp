#include "broadcast_to.hpp"

#include <optional>
#include <string_view>
#include <stdexcept>
#include <array>
#include <utility>
#include <memory>

#include <minitensor/types.hpp>
#include <minitensor/ops.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/tensor_access.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/core/broadcast_shape.hpp"

namespace minitensor
{
    namespace detail
    {
        BroadcastToPrimitive::BroadcastToPrimitive(Shape shape) : shape_(std::move(shape)) {}

        std::string_view BroadcastToPrimitive::name() const noexcept
        {
            return "broadcast_to";
        }

        TensorSpec BroadcastToPrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"broadcast_to requires a single input"};
            }
            const TensorSpec &input = inputs.front();
            if (broadcast_shape(input.shape, shape_) != shape_)
            {
                throw std::invalid_argument{"requested shape is incompatible for broadcasting"};
            }
            return TensorSpec{shape_, input.dtype, input.device};
        }

        std::optional<Layout> BroadcastToPrimitive::try_derive_shared_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const
        {
            return input_layout.broadcasted_to(input_spec.shape, output_spec.shape);
        }

        const Shape &BroadcastToPrimitive::shape() const noexcept
        {
            return shape_;
        }

        bool BroadcastToPrimitive::requires_kernel_support() const noexcept
        {
            return false;
        }
    }

    Tensor broadcast_to(const Tensor &input, Shape shape)
    {
        auto primitive = std::make_unique<detail::BroadcastToPrimitive>(std::move(shape));
        std::array<detail::ValueRef, 1> inputs{detail::TensorAccess::value(input)};
        detail::ValueRef output = detail::apply_operation(std::move(primitive), inputs);
        return detail::TensorAccess::make(std::move(output));
    }
}