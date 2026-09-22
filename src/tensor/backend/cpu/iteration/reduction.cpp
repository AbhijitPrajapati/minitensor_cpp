#include "reduction.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace minitensor::detail::cpu
{
    ReductionPlan::ReductionPlan(
        const Shape &input_shape,
        const Layout &input_layout,
        std::span<const size_type> reduced_axes)
        : input_base_offset_{input_layout.offset()},
          input_numel_{input_shape.numel()}
    {
        if (input_layout.rank() != input_shape.rank())
        {
            throw std::invalid_argument{
                "input layout rank does not match reduction shape rank"};
        }

        std::vector<bool> is_reduced(input_shape.rank(), false);
        for (const size_type axis : reduced_axes)
        {
            if (axis >= input_shape.rank())
            {
                throw std::invalid_argument{"reduction axis is out of range"};
            }
            if (is_reduced[axis])
            {
                throw std::invalid_argument{"reduction axes contain a duplicate"};
            }
            is_reduced[axis] = true;
        }

        outer_dimensions_.reserve(input_shape.rank() - reduced_axes.size());
        reduction_dimensions_.reserve(reduced_axes.size());

        for (size_type axis = 0; axis < input_shape.rank(); ++axis)
        {
            const size_type extent = static_cast<size_type>(input_shape[axis]);
            size_type &count = is_reduced[axis] ? reduction_size_ : output_size_;
            count = checked_product(count, extent);

            // Singleton axes do not affect offsets
            if (extent <= 1)
            {
                continue;
            }

            const stride_type step = input_layout.stride(axis);
            const Dimension dimension{
                extent,
                step,
                step * static_cast<offset_type>(extent - 1)};

            if (is_reduced[axis])
            {
                reduction_dimensions_.push_back(dimension);
            }
            else
            {
                outer_dimensions_.push_back(dimension);
            }
        }

        coalesce_reduction_suffix();
    }

    ReductionPlan::size_type ReductionPlan::input_numel() const noexcept
    {
        return input_numel_;
    }

    ReductionPlan::size_type ReductionPlan::output_size() const noexcept
    {
        return output_size_;
    }

    ReductionPlan::size_type ReductionPlan::reduction_size() const noexcept
    {
        return reduction_size_;
    }

    ReductionPlan::size_type ReductionPlan::checked_product(size_type lhs, size_type rhs)
    {
        if (rhs != 0 && lhs > std::numeric_limits<size_type>::max() / rhs)
        {
            throw std::overflow_error{"reduction iteration size overflow"};
        }
        return lhs * rhs;
    }

    void ReductionPlan::coalesce_reduction_suffix() noexcept
    {
        if (reduction_size_ == 0)
        {
            reduction_run_count_ = 0;
            return;
        }

        if (reduction_dimensions_.empty())
        {
            return;
        }

        const Dimension &innermost = reduction_dimensions_.back();
        reduction_run_size_ = innermost.extent;
        reduction_run_stride_ = innermost.step;
        std::size_t run_axis_start = reduction_dimensions_.size() - 1;

        while (run_axis_start > 0)
        {
            if (reduction_run_size_ >
                static_cast<size_type>(std::numeric_limits<offset_type>::max()))
            {
                break;
            }

            const offset_type signed_run_size =
                static_cast<offset_type>(reduction_run_size_);
            const Dimension &outer = reduction_dimensions_[run_axis_start - 1];
            if (outer.step % signed_run_size != 0 ||
                outer.step / signed_run_size != reduction_run_stride_)
            {
                break;
            }

            reduction_run_size_ *= outer.extent;
            --run_axis_start;
        }

        reduction_dimensions_.resize(run_axis_start);
        reduction_run_count_ = reduction_size_ / reduction_run_size_;
    }

    ReductionPlan::size_type ReductionPlan::ReductionRange::size() const noexcept
    {
        return size_;
    }
}
