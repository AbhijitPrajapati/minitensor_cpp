#include "uniform.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <minitensor/random.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
UniformPrimitive::UniformPrimitive(TensorSpec output_spec, float low, float high, RandomKey key)
    : output_spec_(std::move(output_spec)), low_(low), high_(high), key_(key)
{
    if (!is_floating_point(output_spec_.dtype))
    {
        throw std::invalid_argument{"uniform requires a floating-point dtype"};
    }
    if (!std::isfinite(low_) || !std::isfinite(high_))
    {
        throw std::invalid_argument{"uniform bounds must be finite"};
    }
    if (low_ > high_)
    {
        throw std::invalid_argument{"uniform lower bound cannot exceed upper bound"};
    }
}

std::string_view UniformPrimitive::name() const noexcept
{
    return "uniform";
}

TensorSpec UniformPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (!inputs.empty())
    {
        throw std::invalid_argument{"uniform expects no input tensors"};
    }
    return output_spec_;
}

float UniformPrimitive::low() const noexcept
{
    return low_;
}

float UniformPrimitive::high() const noexcept
{
    return high_;
}

const RandomKey &UniformPrimitive::key() const noexcept
{
    return key_;
}

} // namespace minitensor::detail
