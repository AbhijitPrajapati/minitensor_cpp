#include "minitensor/types.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace minitensor
{
    namespace
    {

        Index checked_numel(const std::vector<Index> &extents)
        {
            Index result = 1;
            for (const auto extent : extents)
            {
                if (extent < 0)
                {
                    throw std::invalid_argument("tensor dimensions must be non-negative");
                }
                if (result != 0 && extent > std::numeric_limits<Index>::max() / result)
                {
                    throw std::overflow_error("tensor element count overflows int64_t");
                }
                result *= extent;
            }
            return result;
        }

        void validate_rank(const std::size_t rank)
        {
            if (static_cast<std::uint64_t>(rank) >
                static_cast<std::uint64_t>(std::numeric_limits<Index>::max()))
            {
                throw std::length_error("tensor rank does not fit int64_t");
            }
        }

        std::size_t as_size(const Index value)
        {
            return static_cast<std::size_t>(value);
        }

    } // namespace

    Shape::Shape(std::initializer_list<Index> extents)
        : Shape(std::vector<Index>(extents)) {}

    Shape::Shape(std::vector<Index> extents)
        : extents_(std::move(extents))
    {
        validate_rank(extents_.size());
        static_cast<void>(checked_numel(extents_));
    }

    Shape::Shape(const size_type rank_value, const Index extent)
        : Shape(std::vector<Index>(rank_value, extent)) {}

    Shape::size_type Shape::size() const noexcept { return extents_.size(); }
    bool Shape::empty() const noexcept { return extents_.empty(); }
    Index Shape::rank() const noexcept { return static_cast<Index>(extents_.size()); }
    Index Shape::numel() const noexcept
    {
        Index result = 1;
        for (const auto extent : extents_)
        {
            result *= extent;
        }
        return result;
    }
    Index Shape::operator[](const size_type dimension) const noexcept { return extents_[dimension]; }

    Shape::const_iterator Shape::begin() const noexcept { return extents_.begin(); }
    Shape::const_iterator Shape::end() const noexcept { return extents_.end(); }
    Shape::const_reverse_iterator Shape::rbegin() const noexcept { return extents_.rbegin(); }
    Shape::const_reverse_iterator Shape::rend() const noexcept { return extents_.rend(); }

    void Shape::require_axis(const Index dimension, const std::string_view operation) const
    {
        if (dimension < 0 || dimension >= rank())
        {
            throw std::out_of_range(std::string(operation) + " dimension is out of range");
        }
    }

    void Shape::require_reshape_compatible(const Shape &other) const
    {
        if (!is_reshape_compatible_with(other))
        {
            throw std::invalid_argument("reshape must preserve the number of elements");
        }
    }

    bool Shape::is_reshape_compatible_with(const Shape &other) const noexcept
    {
        return numel() == other.numel();
    }

    Shape Shape::broadcast_with(const Shape &other) const
    {
        const auto output_rank = std::max(size(), other.size());
        std::vector<Index> output_extents(output_rank, 1);
        const auto lhs_padding = output_rank - size();
        const auto rhs_padding = output_rank - other.size();

        for (std::size_t dimension = 0; dimension < output_rank; ++dimension)
        {
            const auto lhs_extent = dimension < lhs_padding ? Index{1} : extents_[dimension - lhs_padding];
            const auto rhs_extent = dimension < rhs_padding ? Index{1} : other.extents_[dimension - rhs_padding];
            if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
            {
                throw std::invalid_argument("tensor shapes are not broadcast-compatible");
            }
            output_extents[dimension] = lhs_extent == 1 ? rhs_extent : lhs_extent;
        }
        return Shape(std::move(output_extents));
    }

    bool Shape::is_broadcastable_to(const Shape &target) const noexcept
    {
        if (rank() > target.rank())
        {
            return false;
        }

        const auto rank_difference = target.size() - size();
        for (std::size_t dimension = 0; dimension < size(); ++dimension)
        {
            const auto source_extent = extents_[dimension];
            const auto target_extent = target.extents_[rank_difference + dimension];
            if (source_extent != target_extent && source_extent != 1)
            {
                return false;
            }
        }
        return true;
    }

    Shape Shape::reduced(const std::optional<Index> dimension, const bool keepdim) const
    {
        if (!dimension.has_value())
        {
            return keepdim ? Shape(size(), 1) : Shape{};
        }

        require_axis(*dimension, "sum");
        auto output_extents = extents_;
        const auto index = as_size(*dimension);
        if (keepdim)
        {
            output_extents[index] = 1;
        }
        else
        {
            output_extents.erase(output_extents.begin() + static_cast<std::ptrdiff_t>(index));
        }
        return Shape(std::move(output_extents));
    }

    Shape Shape::transposed(const Index dim0, const Index dim1) const
    {
        require_axis(dim0, "transpose");
        require_axis(dim1, "transpose");
        auto result = extents_;
        std::swap(result[as_size(dim0)], result[as_size(dim1)]);
        return Shape(std::move(result));
    }

    Shape Shape::with_extent(const Index dimension, const Index extent) const
    {
        require_axis(dimension, "shape");
        auto result = extents_;
        result[as_size(dimension)] = extent;
        return Shape(std::move(result));
    }

    Shape Shape::with_inserted_axis(const Index dimension, const Index extent) const
    {
        if (dimension < 0 || dimension > rank())
        {
            throw std::out_of_range("inserted dimension is out of range");
        }
        auto result = extents_;
        result.insert(result.begin() + static_cast<std::ptrdiff_t>(dimension), extent);
        return Shape(std::move(result));
    }

} // namespace minitensor
