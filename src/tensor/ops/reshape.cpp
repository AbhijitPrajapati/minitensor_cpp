#include "reshape.hpp"

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

namespace minitensor
{
    namespace detail
    {
        ReshapePrimitive::ReshapePrimitive(Shape shape) : shape_(std::move(shape)) {}

        std::string_view ReshapePrimitive::name() const noexcept
        {
            return "reshape";
        }

        TensorSpec ReshapePrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"reshape requires a single input"};
            }
            const TensorSpec &input = inputs.front();
            if (input.shape.numel() != shape_.numel())
            {
                throw std::invalid_argument{" requested shape is incompatible "};
            }
            return TensorSpec{shape_, input.dtype, input.device};
        }

        std::optional<Layout> ReshapePrimitive::try_derive_shared_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const
        {
            return input_layout.try_reshape(input_spec.shape, output_spec.shape);
        }

        const Shape &ReshapePrimitive::shape() const noexcept
        {
            return shape_;
        }
    }

    Tensor reshape(const Tensor &input, Shape shape)
    {
        auto primitive = std::make_unique<detail::ReshapePrimitive>(std::move(shape));
        std::array<detail::ValueRef, 1> inputs{detail::TensorAccess::value(input)};
        detail::ValueRef output = detail::apply_operation(std::move(primitive), inputs);
        return detail::TensorAccess::make(std::move(output));
    }
}