#include "reduction.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace minitensor::detail::cuda
{
ReductionPlan::ReductionPlan(const Shape &shape, const Layout &layout, std::span<const size_type> axes)
{
    if (shape.rank() != layout.rank())
    {
        throw std::invalid_argument{"input layout rank does not match reduction shape rank"};
    }
    std::vector<bool> reduced(shape.rank(), false);
    for (const auto axis : axes)
    {
        if (axis >= shape.rank() || reduced[axis])
        {
            throw std::invalid_argument{"reduction axes must be unique and in range"};
        }
        reduced[axis] = true;
    }
    std::vector<Extent> outer_extents;
    std::vector<Extent> reduction_extents;
    std::vector<Layout::stride_type> outer_strides;
    std::vector<Layout::stride_type> reduction_strides;
    outer_extents.reserve(shape.rank() - axes.size());
    outer_strides.reserve(shape.rank() - axes.size());
    reduction_extents.reserve(axes.size());
    reduction_strides.reserve(axes.size());
    for (size_type axis = 0; axis < shape.rank(); ++axis)
    {
        if (!std::in_range<size_type>(shape[axis]))
        {
            throw std::overflow_error{"reduction extent is not representable as size_t"};
        }
        (reduced[axis] ? reduction_extents : outer_extents).push_back(shape[axis]);
        (reduced[axis] ? reduction_strides : outer_strides).push_back(layout.stride(axis));
    }
    outer_shape_ = Shape{std::move(outer_extents)};
    reduction_shape_ = Shape{std::move(reduction_extents)};
    outer_layout_ = Layout{std::move(outer_strides), layout.offset()};
    reduction_layout_ = Layout{std::move(reduction_strides)};
}
} // namespace minitensor::detail::cuda
