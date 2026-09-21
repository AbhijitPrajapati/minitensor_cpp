#include "reduction_common.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <minitensor/ops/manipulation.hpp>

#include "tensor/core/axis.hpp"

namespace minitensor::detail
{
    std::vector<Shape::size_type> normalize_reduction_axes(
        std::span<const Axis> axes,
        Shape::size_type input_rank)
    {
        if (axes.size() > input_rank)
        {
            throw std::invalid_argument{
                "reduction cannot reduce more unique axes than the input rank"};
        }

        std::vector<Shape::size_type> normalized_axes;
        normalized_axes.reserve(axes.size());
        for (const Axis axis : axes)
        {
            normalized_axes.push_back(normalize_axis(axis, input_rank));
        }
        std::ranges::sort(normalized_axes);
        if (std::ranges::adjacent_find(normalized_axes) != normalized_axes.end())
        {
            throw std::invalid_argument{"reduction axes cannot contain duplicates"};
        }
        return normalized_axes;
    }

    TensorSpec infer_reduction(
        std::span<const TensorSpec> inputs,
        std::span<const Shape::size_type> axes,
        Shape::size_type input_rank,
        bool keep_dim,
        std::string_view operation_name,
        bool accepts_empty_reduction)
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument{
                std::string{operation_name} + " requires a single input"};
        }

        const TensorSpec &input = inputs.front();
        if (input.shape.rank() != input_rank)
        {
            throw std::invalid_argument{
                "reduction primitive was normalized for a different input rank"};
        }

        if (!accepts_empty_reduction)
        {
            for (const Shape::size_type axis : axes)
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
        if (keep_dim)
        {
            const auto input_dimensions = input.shape.dimensions();
            output_extents.assign(input_dimensions.begin(), input_dimensions.end());
            for (const Shape::size_type axis : axes)
            {
                output_extents[axis] = Extent{1};
            }
        }
        else
        {
            output_extents.reserve(input.shape.rank() - axes.size());
            auto reduced_axis = axes.begin();
            for (Shape::size_type axis = 0; axis < input.shape.rank(); ++axis)
            {
                if (reduced_axis != axes.end() && *reduced_axis == axis)
                {
                    ++reduced_axis;
                    continue;
                }
                output_extents.push_back(input.shape[axis]);
            }
        }

        return TensorSpec{Shape{std::move(output_extents)}, input.dtype, input.device};
    }

    Tensor broadcast_reduced_tensor(
        const Tensor &reduced,
        const Shape &input_shape,
        std::span<const Shape::size_type> axes,
        bool keep_dim)
    {
        Tensor expanded = reduced;
        if (!axes.empty() && !keep_dim)
        {
            std::vector<Extent> expanded_dimensions(
                input_shape.dimensions().begin(), input_shape.dimensions().end());
            for (const Shape::size_type axis : axes)
            {
                expanded_dimensions[axis] = Extent{1};
            }
            expanded = reshape(reduced, Shape{std::move(expanded_dimensions)});
        }

        return expanded.shape() == input_shape
                   ? expanded
                   : broadcast_to(expanded, input_shape);
    }
}
