#include "max.hpp"

namespace minitensor::detail
{
    MaxPrimitive::MaxPrimitive(
        std::span<const Axis> axes,
        Shape::size_type input_rank,
        bool keep_dim)
        : reduction_(axes, input_rank, keep_dim) {}

    std::string_view MaxPrimitive::name() const noexcept
    {
        return "max";
    }

    TensorSpec MaxPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        return reduction_.infer(inputs, name(), false);
    }

    const std::vector<Shape::size_type> &MaxPrimitive::axes() const noexcept
    {
        return reduction_.axes();
    }

    bool MaxPrimitive::keep_dim() const noexcept
    {
        return reduction_.keep_dim();
    }
}
