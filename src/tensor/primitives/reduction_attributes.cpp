#include "reduction_attributes.hpp"

#include <algorithm>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <minitensor/ops.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    ReductionAttributes::ReductionAttributes(
        std::span<const Axis> axes,
        Shape::size_type input_rank,
        bool keep_dim)
        : input_rank_(input_rank), keep_dim_(keep_dim)
    {
        if (axes.size() > input_rank_)
        {
            throw std::invalid_argument{
                "reduction cannot reduce more unique axes than the input rank"};
        }

        axes_.reserve(axes.size());
        for (const Axis axis : axes)
        {
            axes_.push_back(normalize_axis(axis, input_rank_));
        }
        std::ranges::sort(axes_);
        if (std::ranges::adjacent_find(axes_) != axes_.end())
        {
            throw std::invalid_argument{"reduction axes cannot contain duplicates"};
        }
    }

    TensorSpec ReductionAttributes::infer(
        std::span<const TensorSpec> inputs,
        std::string_view operation_name,
        bool accepts_empty_reduction) const
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument{
                std::string{operation_name} + " requires a single input"};
        }

        const TensorSpec &input = inputs.front();
        if (input.shape.rank() != input_rank_)
        {
            throw std::invalid_argument{
                "reduction primitive was normalized for a different input rank"};
        }

        if (!accepts_empty_reduction)
        {
            for (const Shape::size_type axis : axes_)
            {
                if (input.shape[axis] == Extent{0})
                {
                    throw std::invalid_argument{
                        std::string{operation_name} +
                        " cannot reduce an axis with extent zero"};
                }
            }
        }

        std::vector<Extent> output_extents;
        if (keep_dim_)
        {
            const auto input_dimensions = input.shape.dimensions();
            output_extents.assign(input_dimensions.begin(), input_dimensions.end());
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

    const std::vector<Shape::size_type> &ReductionAttributes::axes() const noexcept
    {
        return axes_;
    }

    bool ReductionAttributes::keep_dim() const noexcept
    {
        return keep_dim_;
    }

    std::size_t ReductionAttributes::reduction_size(const Shape &input_shape) const
    {
        if (input_shape.rank() != input_rank_)
        {
            throw std::invalid_argument{"reduction size requested for a different input rank"};
        }

        std::size_t size = 1;
        for (const Shape::size_type axis : axes_)
        {
            const Extent extent = input_shape[axis];
            if (extent == Extent{0})
            {
                return 0;
            }

            const auto converted = static_cast<std::size_t>(extent);
            if (size > std::numeric_limits<std::size_t>::max() / converted)
            {
                throw std::overflow_error{"reduction size overflow"};
            }
            size *= converted;
        }
        return size;
    }

    Tensor ReductionAttributes::expand_reduced_tensor(
        const Tensor &reduced,
        const Shape &input_shape) const
    {
        if (axes_.empty() || keep_dim_)
        {
            return reduced;
        }

        std::vector<Extent> expanded_dimensions(
            input_shape.dimensions().begin(), input_shape.dimensions().end());
        for (const Shape::size_type axis : axes_)
        {
            expanded_dimensions[axis] = Extent{1};
        }
        return reshape(reduced, Shape{std::move(expanded_dimensions)});
    }

    Tensor ReductionAttributes::broadcast_reduced_tensor(
        const Tensor &reduced,
        const Shape &input_shape) const
    {
        Tensor expanded = expand_reduced_tensor(reduced, input_shape);
        return expanded.shape() == input_shape
                   ? expanded
                   : broadcast_to(expanded, input_shape);
    }

}
