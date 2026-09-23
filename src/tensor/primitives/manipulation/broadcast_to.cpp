#include "broadcast_to.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/core/shape_inference.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
    BroadcastToPrimitive::BroadcastToPrimitive(Shape shape) : shape_(std::move(shape))
    {
    }

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

    std::vector<std::optional<Tensor>> BroadcastToPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 1)
        {
            throw std::logic_error{"broadcast_to VJP expects 1 input"};
        }
        return {reduce_to_shape(output_cotangent, inputs.front().shape())};
    }
}
