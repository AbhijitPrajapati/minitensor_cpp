#include "reduction.hpp"

#include <span>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
Shape reduction_shape(const Shape &input_shape, std::span<const Shape::size_type> reduced_axes, bool keep_dim)
{
    std::vector<Extent> output_extents;
    if (keep_dim)
    {
        output_extents.assign(input_shape.begin(), input_shape.end());
        for (const Shape::size_type axis : reduced_axes)
        {
            output_extents[axis] = Extent{1};
        }
    }
    else
    {
        output_extents.reserve(input_shape.rank() - reduced_axes.size());
        auto reduced_axis = reduced_axes.begin();
        for (Shape::size_type axis = 0; axis < input_shape.rank(); ++axis)
        {
            if (reduced_axis != reduced_axes.end() && *reduced_axis == axis)
            {
                ++reduced_axis;
                continue;
            }
            output_extents.push_back(input_shape[axis]);
        }
    }
    return Shape(std::move(output_extents));
}
} // namespace minitensor::detail