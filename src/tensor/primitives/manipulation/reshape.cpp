#include "reshape.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
ReshapePrimitive::ReshapePrimitive(Shape shape) : shape_(std::move(shape))
{
}

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

// Reshape may or may not allocate a new buffer depending on the input layout.
std::optional<Layout> ReshapePrimitive::try_derive_shared_layout(const TensorSpec &input_spec,
                                                                 const Layout &input_layout,
                                                                 const TensorSpec &output_spec) const
{
    return input_layout.try_reshape(input_spec.shape, output_spec.shape);
}

const Shape &ReshapePrimitive::shape() const noexcept
{
    return shape_;
}

} // namespace minitensor::detail
