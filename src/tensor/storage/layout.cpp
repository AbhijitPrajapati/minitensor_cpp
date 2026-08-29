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
}
