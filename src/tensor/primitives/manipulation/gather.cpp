#include "gather.hpp"

#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
GatherPrimitive::GatherPrimitive(Axis axis, Shape::size_type input_rank)
    : axis_(normalize_axis(axis, input_rank)), input_rank_(input_rank)
{
}

std::string_view GatherPrimitive::name() const noexcept
{
    return "gather";
}

Shape::size_type GatherPrimitive::axis() const noexcept
{
    return axis_;
}

TensorSpec GatherPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument{"gather expects 2 input tensors"};
    }

    const TensorSpec &source = inputs[0];
    const TensorSpec &indices = inputs[1];

    if (source.shape.rank() != input_rank_ || indices.shape.rank() != input_rank_)
    {
        throw std::invalid_argument{"gather primitive was normalized for a different input rank"};
    }

    if (indices.dtype != DType::Int64)
    {
        throw std::invalid_argument{"gather requires Int64 indices"};
    }

    if (source.device != indices.device)
    {
        throw std::invalid_argument{"gather requires input tensors on the same device"};
    }

    for (Shape::size_type axis = 0; axis < input_rank_; ++axis)
    {
        if (axis == axis_)
        {
            continue;
        }

        // Extents other than the gather axis do not necessarily have
        // to be equal; The index extent just cannot be greater than
        // the source extent. If the index extent is less than the source
        // extent, then the missing slices will simply remain unsampled.
        if (indices.shape[axis] > source.shape[axis])
        {
            throw std::invalid_argument{"gather requires compatible index and source shapes"};
        }
    }

    return TensorSpec{indices.shape, source.dtype, source.device};
}

GatherScatterAddPrimitive::GatherScatterAddPrimitive(Shape target_shape, Shape::size_type axis)
    : target_shape_(std::move(target_shape)), axis_(axis)
{
    if (axis_ >= target_shape_.rank())
    {
        throw std::out_of_range{"gather scatter add axis is out of range"};
    }
}

std::string_view GatherScatterAddPrimitive::name() const noexcept
{
    return "gather_scatter_add";
}

Shape::size_type GatherScatterAddPrimitive::axis() const noexcept
{
    return axis_;
}

const Shape &GatherScatterAddPrimitive::target_shape() const noexcept
{
    return target_shape_;
}

// GatherScatterAdd takes in the updates tensor (the output cotangent
// in reverse VJP) as well as the original indices used for the gather.
TensorSpec GatherScatterAddPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument{"gather scatter add expects 2 input tensors"};
    }

    const TensorSpec &updates = inputs[0];
    const TensorSpec &indices = inputs[1];

    if (updates.shape != indices.shape)
    {
        throw std::invalid_argument{"gather scatter add requires updates and indices to have the same shape"};
    }

    if (updates.shape.rank() != target_shape_.rank() || indices.shape.rank() != target_shape_.rank())
    {
        throw std::invalid_argument{"gather scatter add primitive was normalized for a different target shape"};
    }

    if (indices.dtype != DType::Int64)
    {
        throw std::invalid_argument{"gather scatter add requires Int64 indices"};
    }

    if (!is_floating_point(updates.dtype))
    {
        throw std::invalid_argument{"gather scatter add requires floating point updates"};
    }

    if (updates.device != indices.device)
    {
        throw std::invalid_argument{"gather scatter add requires updates and indices on the same device"};
    }

    for (Shape::size_type axis = 0; axis < target_shape_.rank(); ++axis)
    {
        if (axis == axis_)
        {
            continue;
        }

        if (indices.shape[axis] > target_shape_[axis])
        {
            throw std::invalid_argument{"gather scatter add requires compatible index shape"};
        }
    }

    return TensorSpec{target_shape_, updates.dtype, updates.device};
}

} // namespace minitensor::detail
