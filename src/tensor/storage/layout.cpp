#include "layout.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
    Layout::Layout(std::vector<stride_type> strides, offset_type offset) : strides_(std::move(strides)), offset_(offset)
    {
        if (offset_ < 0)
        {
            throw std::invalid_argument{"offset cannot be negative"};
        }
    }

    Layout Layout::contiguous(const Shape &shape)
    {
        std::vector<stride_type> strides(shape.rank(), stride_type{1});

        stride_type running_stride = 1;

        for (size_type i = shape.rank(); i > 0; --i)
        {
            const size_type axis = i - 1;
            const Extent extent = shape[axis];

            strides[axis] = running_stride;

            if (extent < 2)
            {
                continue;
            }

            if (running_stride > std::numeric_limits<stride_type>::max() / extent)
            {
                throw std::overflow_error{"contiguous stride overflow"};
            }
            running_stride *= extent;
        }
        return Layout{std::move(strides), offset_type{0}};
    }

    Layout::size_type Layout::rank() const noexcept
    {
        return strides_.size();
    }

    Layout::stride_type Layout::stride(size_type axis) const noexcept
    {
        return strides_[axis];
    }

    Layout::offset_type Layout::offset() const noexcept
    {
        return offset_;
    }

    bool Layout::is_contiguous(const Shape &shape) const noexcept
    {
        if (rank() != shape.rank())
        {
            return false;
        }
        if (shape.numel() == 0)
        {
            return true;
        }

        stride_type expected = 1;
        for (size_type i = shape.rank(); i > 0; --i)
        {
            const size_type axis = i - 1;
            const Extent extent = shape[axis];
            if (extent == 1)
            {
                continue;
            }
            if (strides_[axis] != expected)
            {
                return false;
            }
            if (expected > std::numeric_limits<stride_type>::max() / extent)
            {
                return false;
            }
            expected *= extent;
        }
        return true;
    }

    std::span<const Layout::stride_type> Layout::strides() const noexcept
    {
        return strides_;
    }

    Layout Layout::broadcasted_to(const Shape &source_shape, const Shape &target_shape) const
    {
        if (source_shape.rank() != rank())
        {
            throw std::invalid_argument{"source shape and layout ranks do not match"};
        }
        if (source_shape.rank() > target_shape.rank())
        {
            throw std::invalid_argument{"source shape rank exceeds target shape rank"};
        }

        std::vector<stride_type> output_strides(target_shape.rank(), stride_type{0});
        const std::size_t rank_diff = target_shape.rank() - source_shape.rank();

        for (std::size_t source_axis = 0; source_axis < source_shape.rank(); ++source_axis)
        {
            const std::size_t target_axis = rank_diff + source_axis;
            const Extent source_extent = source_shape[source_axis];
            const Extent target_extent = target_shape[target_axis];
            if (source_extent == target_extent)
            {
                output_strides[target_axis] = strides_[source_axis];
            }
            else if (source_extent == 1)
            {
                output_strides[target_axis] = 0;
            }
            else
            {
                throw std::invalid_argument{"cannot broadcast to target shape"};
            }
        }

        return Layout(std::move(output_strides), offset_);
    }
}
