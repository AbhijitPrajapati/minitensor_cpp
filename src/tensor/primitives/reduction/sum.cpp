#include "sum.hpp"

#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
SumPrimitive::SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim)
    : input_rank_(input_rank), keep_dim_(keep_dim)
{
    if (axes.size() > input_rank_)
    {
        throw std::invalid_argument{"sum cannot reduce more unique axes than the input rank"};
    }

    axes_.reserve(axes.size());
    for (const Axis axis : axes)
    {
        axes_.push_back(normalize_axis(axis, input_rank_));
    }
    std::ranges::sort(axes_);
    if (std::ranges::adjacent_find(axes_) != axes_.end())
    {
        throw std::invalid_argument{"sum axes cannot contain duplicates"};
    }
}

std::string_view SumPrimitive::name() const noexcept
{
    return "sum";
}

TensorSpec SumPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument{"sum requires a single input"};
    }

    const TensorSpec &input = inputs.front();
    if (input.shape.rank() != input_rank_)
    {
        throw std::invalid_argument{"sum primitive was normalized for a different input rank"};
    }
    if (!is_floating_point(input.dtype))
    {
        throw std::invalid_argument{"sum requires a floating-point input"};
    }

    std::vector<Extent> output_extents;
    if (keep_dim_)
    {
        output_extents.assign(input.shape.begin(), input.shape.end());
        for (const Shape::size_type axis : axes_)
        {
            output_extents[axis] = Extent{1};
        }
    }
    else
    {
        output_extents.reserve(input.shape.rank() - axes_.size());
        auto reduced_axis = axes_.begin();
        for (Shape::size_type axis = 0; axis < input.shape.rank(); ++axis)
        {
            if (reduced_axis != axes_.end() && *reduced_axis == axis)
            {
                ++reduced_axis;
                continue;
            }
            output_extents.push_back(input.shape[axis]);
        }
    }

    return TensorSpec{Shape{std::move(output_extents)}, input.dtype, input.device};
}

const std::vector<Shape::size_type> &SumPrimitive::axes() const noexcept
{
    return axes_;
}

bool SumPrimitive::keep_dim() const noexcept
{
    return keep_dim_;
}

} // namespace minitensor::detail
