#include "normal.hpp"

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
NormalPrimitive::NormalPrimitive(TensorSpec output_spec, float mean, float std_dev, RandomKey key)
    : output_spec_(std::move(output_spec)), mean_(mean), std_dev_(std_dev), key_(key)
{
    if (!is_floating_point(output_spec_.dtype))
    {
        throw std::invalid_argument{"normal requires a floating-point dtype"};
    }
    if (!std::isfinite(mean_) || !std::isfinite(std_dev_))
    {
        throw std::invalid_argument{"normal parameters must be finite"};
    }
    if (std_dev_ < 0.0F)
    {
        throw std::invalid_argument{"normal standard deviation cannot be negative"};
    }
}

std::string_view NormalPrimitive::name() const noexcept
{
    return "normal";
}

TensorSpec NormalPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (!inputs.empty())
    {
        throw std::invalid_argument{"normal expects no input tensors"};
    }
    return output_spec_;
}

float NormalPrimitive::mean() const noexcept
{
    return mean_;
}

float NormalPrimitive::std_dev() const noexcept
{
    return std_dev_;
}

const RandomKey &NormalPrimitive::key() const noexcept
{
    return key_;
}

} // namespace minitensor::detail
