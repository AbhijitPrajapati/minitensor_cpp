#include "square_root.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
std::string_view SquareRootPrimitive::name() const noexcept
{
    return "sqrt";
}

TensorSpec SquareRootPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument{"sqrt expects 1 input tensor"};
    }
    if (!is_floating_point(inputs.front().dtype))
    {
        throw std::invalid_argument{"sqrt requires a floating-point input"};
    }
    return inputs.front();
}

} // namespace minitensor::detail
