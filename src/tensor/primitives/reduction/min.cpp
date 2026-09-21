#include "min.hpp"

#include "reduction_common.hpp"

namespace minitensor::detail
{
    MinPrimitive::MinPrimitive(
        std::span<const Axis> axes,
        Shape::size_type input_rank,
        bool keep_dim)
        : axes_(normalize_reduction_axes(axes, input_rank)),
          input_rank_(input_rank),
          keep_dim_(keep_dim) {}

    std::string_view MinPrimitive::name() const noexcept
    {
        return "min";
    }

    TensorSpec MinPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        return infer_reduction(inputs, axes_, input_rank_, keep_dim_, name(), false);
    }

    const std::vector<Shape::size_type> &MinPrimitive::axes() const noexcept
    {
        return axes_;
    }

    bool MinPrimitive::keep_dim() const noexcept
    {
        return keep_dim_;
    }
}
