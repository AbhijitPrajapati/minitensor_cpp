#include "exponential.hpp"

#include <span>
#include <stdexcept>
#include <string_view>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
std::string_view ExponentialPrimitive::name() const noexcept
{
    return "exp";
}

TensorSpec ExponentialPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument{"exp expects 1 input tensor"};
    }
    if (!is_floating_point(inputs.front().dtype))
    {
        throw std::invalid_argument{"exp requires a floating-point input"};
    }
    return inputs.front();
}

} // namespace minitensor::detail
