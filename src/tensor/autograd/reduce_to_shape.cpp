#include "reduce_to_shape.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>
#include <minitensor/ops.hpp>

namespace minitensor::detail
{
    Tensor reduce_to_shape(const Tensor &cotangent, const Shape &target_shape)
    {
        const Shape &cotangent_shape = cotangent.shape();

        if (cotangent_shape == target_shape)
        {
            return cotangent;
        }

        if (target_shape.rank() > cotangent_shape.rank())
        {
            throw std::invalid_argument{"cannot reduce a tensor to a higher rank shape"};
        }

        if (cotangent_shape.rank() > static_cast<Shape::size_type>(std::numeric_limits<Axis>::max()))
        {
            throw std::overflow_error{"rank exceeds maximum axis"};
        }

        const Shape::size_type num_leading = cotangent_shape.rank() - target_shape.rank();
        std::vector<Axis> reduction_axes;

        for (Shape::size_type axis = 0; axis < num_leading; ++axis)
        {
            if (cotangent_shape[axis] != Extent{1})
            {
                reduction_axes.push_back(static_cast<Axis>(axis));
            }
        }

        for (Shape::size_type target_axis = 0; target_axis < target_shape.rank(); ++target_axis)
        {
            const Shape::size_type cotangent_axis = num_leading + target_axis;

            const Extent cotangent_extent = cotangent_shape[cotangent_axis];
            const Extent target_extent = target_shape[target_axis];

            if (cotangent_extent == target_extent)
            {
                continue;
            }

            if (target_extent != Extent{ 1 })
            {
                throw std::invalid_argument{"target shape was not broadcast to cotangent shape"};
            }

            reduction_axes.push_back(static_cast<Axis>(cotangent_axis));
        }

        Tensor reduced = reduction_axes.empty() ? cotangent : sum(cotangent, reduction_axes, true);
        return reduced.shape() == target_shape ? reduced : reshape(reduced, target_shape);
    }
}
