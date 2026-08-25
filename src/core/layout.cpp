#include "core/layout.hpp"
#include "core/shape.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace minitensor::detail
{

    namespace
    {
        // Checks for overflow before multiplying
        Index checked_multiply(const Index lhs, const Index rhs)
        {
            if (lhs < 0 || rhs < 0)
            {
                throw std::invalid_argument("tensor dimensions, strides, and offsets must be non-negative");
            }
            if (lhs != 0 && rhs > std::numeric_limits<Index>::max() / lhs)
            {
                throw std::overflow_error("tensor size or stride overflows int64_t");
            }
            return lhs * rhs;
        }

        // Checks for overflow before adding
        Index checked_add(const Index lhs, const Index rhs)
        {
            if (lhs < 0 || rhs < 0)
            {
                throw std::invalid_argument("tensor dimensions, strides, and offsets must be non-negative");
            }
            if (rhs > std::numeric_limits<Index>::max() - lhs)
            {
                throw std::overflow_error("tensor storage offset overflows int64_t");
            }
            return lhs + rhs;
        }

    } // namespace

    Layout Layout::contiguous(Shape shape, const Index offset)
    {
        static_cast<void>(shape::numel(shape));
        Strides strides(shape.size());
        std::exclusive_scan(
            shape.rbegin(), shape.rend(), strides.rbegin(), Index{1}, checked_multiply);
        return Layout(std::move(shape), std::move(strides), offset);
    }

    Layout::Layout(Shape shape, Strides strides, const Index offset)
        : shape_(std::move(shape)),
          strides_(std::move(strides)),
          offset_(offset),
          numel_(shape::numel(shape_))
    {
        if (shape_.size() != strides_.size())
        {
            throw std::invalid_argument("shape and strides must have the same rank");
        }
        if (offset_ < 0 || std::ranges::any_of(strides_, [](const Index stride)
                                               { return stride < 0; }))
        {
            throw std::invalid_argument("layout strides and offset must be non-negative");
        }
    }

    const Shape &Layout::shape() const noexcept { return shape_; }
    const Strides &Layout::strides() const noexcept { return strides_; }
    Index Layout::offset() const noexcept { return offset_; }
    Index Layout::rank() const noexcept { return shape::rank(shape_); }
    Index Layout::numel() const noexcept { return numel_; }

    bool Layout::is_contiguous() const noexcept
    {
        if (numel() == 0)
        {
            return true;
        }

        Index expected_stride = 1;
        for (Index i = rank(); i-- > 0;)
        {
            const auto index = static_cast<std::size_t>(i);
            if (shape_[index] == 1)
            {
                continue;
            }
            if (strides_[index] != expected_stride)
            {
                return false;
            }
            expected_stride *= shape_[index];
        }
        return true;
    }

    Index Layout::maximum_offset() const
    {
        if (numel() == 0)
        {
            return offset_;
        }
        return std::transform_reduce(
            shape_.begin(),
            shape_.end(),
            strides_.begin(),
            offset_,
            checked_add,
            [](const Index dim, const Index stride)
            {
                return checked_multiply(dim - 1, stride);
            });
    }

    Layout Layout::transposed(const Index dim0, const Index dim1) const
    {
        auto result_shape = shape::transpose(shape_, dim0, dim1);
        auto result_strides = strides_;
        std::swap(
            result_strides[static_cast<std::size_t>(dim0)],
            result_strides[static_cast<std::size_t>(dim1)]);
        return Layout(std::move(result_shape), std::move(result_strides), offset_);
    }

    Layout Layout::sliced(
        const Index dim,
        const Index start,
        const Index stop,
        const Index step) const
    {
        shape::require_axis(shape_, dim, "slice");
        if (step <= 0)
        {
            throw std::invalid_argument("slice step must be positive");
        }

        const auto dimension = static_cast<std::size_t>(dim);
        if (start < 0 || stop < 0 || start > stop || stop > shape_[dimension])
        {
            throw std::out_of_range("slice bounds are outside the selected dimension");
        }

        const auto span = stop - start;
        const auto slice_extent = span == 0 ? Index{0} : 1 + (span - 1) / step;
        auto result_shape = shape::replace_extent(shape_, dim, slice_extent);
        auto result_strides = strides_;
        result_strides[dimension] = checked_multiply(result_strides[dimension], step);

        auto result_offset = offset_;
        if (span != 0)
        {
            Coordinates first_coordinates(shape_.size(), 0);
            first_coordinates[dimension] = start;
            result_offset = offset_from_coordinates(first_coordinates);
        }
        return Layout(std::move(result_shape), std::move(result_strides), result_offset);
    }

    Layout Layout::reshaped(Shape shape) const
    {
        shape::require_reshape_compatible(shape_, shape);
        if (!is_contiguous())
        {
            throw std::logic_error("reshape view requires a contiguous layout");
        }
        return Layout::contiguous(std::move(shape), offset_);
    }

    Layout Layout::broadcast_to(const Shape &shape) const
    {
        if (!shape::is_broadcastable_to(shape_, shape))
        {
            throw std::invalid_argument("cannot broadcast layout to requested shape");
        }

        Strides result_strides(shape.size(), 0);
        const auto output_rank = shape::rank(shape);
        const auto rank_difference = output_rank - rank();
        for (Index dim = rank_difference; dim < output_rank; ++dim)
        {
            const auto result_index = static_cast<std::size_t>(dim);
            const auto source_index = static_cast<std::size_t>(dim - rank_difference);
            if (shape_[source_index] == shape[result_index])
            {
                result_strides[result_index] = strides_[source_index];
            }
        }
        return Layout(shape, std::move(result_strides), offset_);
    }

    Index Layout::offset_from_coordinates(const std::span<const Index> coordinates) const
    {
        if (coordinates.size() != shape_.size())
        {
            throw std::invalid_argument("coordinate rank does not match layout rank");
        }
        auto result = offset_;
        for (std::size_t dim = 0; dim < coordinates.size(); ++dim)
        {
            if (coordinates[dim] < 0 || coordinates[dim] >= shape_[dim])
            {
                throw std::out_of_range("tensor coordinate is outside the layout");
            }
            result = checked_add(
                result, checked_multiply(coordinates[dim], strides_[dim]));
        }
        return result;
    }

} // namespace minitensor::detail
