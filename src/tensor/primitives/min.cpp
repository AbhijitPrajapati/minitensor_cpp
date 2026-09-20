#include "min.hpp"

namespace minitensor::detail
{
    MinPrimitive::MinPrimitive(
        std::span<const Axis> axes,
        Shape::size_type input_rank,
        bool keep_dim)
        : reduction_(axes, input_rank, keep_dim) {}

    std::string_view MinPrimitive::name() const noexcept
    {
        return "min";
    }

    TensorSpec MinPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        return reduction_.infer(inputs, name(), false);
    }

    const std::vector<Shape::size_type> &MinPrimitive::axes() const noexcept
    {
        return reduction_.axes();
    }

    bool MinPrimitive::keep_dim() const noexcept
    {
        return reduction_.keep_dim();
    }
}
