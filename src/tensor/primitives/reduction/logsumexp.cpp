#include "logsumexp.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/reduction.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
LogSumExpPrimitive::LogSumExpPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim)
    : input_rank_(input_rank), keep_dim_(keep_dim)
{
    if (axes.size() > input_rank_)
    {
        throw std::invalid_argument{"log sum exp cannot reduce more unique axes than the input rank"};
    }

    axes_.reserve(axes.size());
    for (const Axis axis : axes)
    {
        axes_.push_back(normalize_axis(axis, input_rank_));
    }
    std::ranges::sort(axes_);
    if (std::ranges::adjacent_find(axes_) != axes_.end())
    {
        throw std::invalid_argument{"log sum exp axes cannot contain duplicates"};
    }
}

std::string_view LogSumExpPrimitive::name() const noexcept
{
    return "log_sum_exp";
}

TensorSpec LogSumExpPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument{"log sum exp requires a single input"};
    }

    const TensorSpec &input = inputs.front();
    if (input.shape.rank() != input_rank_)
    {
        throw std::invalid_argument{"log sum exp primitive was normalized for a different input rank"};
    }
    if (!is_floating_point(input.dtype))
    {
        throw std::invalid_argument{"log sum exp requires a floating-point input"};
    }
    return TensorSpec{reduction_shape(input.shape, axes_, keep_dim_), input.dtype, input.device};
}

const std::vector<Shape::size_type> &LogSumExpPrimitive::axes() const noexcept
{
    return axes_;
}

bool LogSumExpPrimitive::keep_dim() const noexcept
{
    return keep_dim_;
}

} // namespace minitensor::detail
