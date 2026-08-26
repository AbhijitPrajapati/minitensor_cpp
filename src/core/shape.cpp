#include "core/shape.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace minitensor::detail::shape
{
    namespace
    {

        Index checked_numel(const Shape &shape)
        {
            Index result = 1;
            for (const auto extent : shape)
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

        void validate_shape(const Shape &shape)
        {
            validate_rank(shape.size());
            checked_numel(shape);
        }

        void require_axis_in_range(
            const Index shape_rank,
            const Index dimension,
            const std::string_view operation)
        {
            if (dimension < 0 || dimension >= shape_rank)
            {
                throw std::out_of_range(
                    std::string(operation) + " dimension is out of range");
            }
        }

        std::size_t as_size(const Index value)
        {
            return static_cast<std::size_t>(value);
        }

    } // namespace

    Index rank(const Shape &shape) noexcept
    {
        return static_cast<Index>(shape.size());
    }

    Index numel(const Shape &shape)
    {
        validate_rank(shape.size());
        return checked_numel(shape);
    }

    void require_axis(
        const Shape &shape,
        const Index dimension,
        const std::string_view operation)
    {
        validate_shape(shape);
        require_axis_in_range(rank(shape), dimension, operation);
    }

    void require_reshape_compatible(const Shape &lhs, const Shape &rhs)
    {
        if (numel(lhs) != numel(rhs))
        {
            throw std::invalid_argument("reshape must preserve the number of elements");
        }
    }

    Shape broadcast(const Shape &lhs, const Shape &rhs)
    {
        validate_shape(lhs);
        validate_shape(rhs);
        const auto output_rank = std::max(lhs.size(), rhs.size());
        std::vector<Index> output_extents(output_rank, 1);
        const auto lhs_padding = output_rank - lhs.size();
        const auto rhs_padding = output_rank - rhs.size();

        for (std::size_t dimension = 0; dimension < output_rank; ++dimension)
        {
            const auto lhs_extent = dimension < lhs_padding ? Index{1} : lhs[dimension - lhs_padding];
            const auto rhs_extent = dimension < rhs_padding ? Index{1} : rhs[dimension - rhs_padding];
            if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
            {
                throw std::invalid_argument("tensor shapes are not broadcast-compatible");
            }
            output_extents[dimension] = lhs_extent == 1 ? rhs_extent : lhs_extent;
        }
        validate_shape(output_extents);
        return output_extents;
    }

    bool is_broadcastable_to(const Shape &source, const Shape &target)
    {
        validate_shape(source);
        validate_shape(target);
        if (source.size() > target.size())
        {
            return false;
        }

        const auto rank_difference = target.size() - source.size();
        for (std::size_t dimension = 0; dimension < source.size(); ++dimension)
        {
            const auto source_extent = source[dimension];
            const auto target_extent = target[rank_difference + dimension];
            if (source_extent != target_extent && source_extent != 1)
            {
                return false;
            }
        }
        return true;
    }

    Shape reduce(const Shape &shape, std::optional<Index> dimension, bool keepdim)
    {
        if (!dimension.has_value())
        {
            validate_shape(shape);
            return keepdim ? Shape(shape.size(), 1) : Shape{};
        }

        require_axis(shape, *dimension, "sum");
        auto output_extents = shape;
        const auto index = as_size(*dimension);
        if (keepdim)
        {
            output_extents[index] = 1;
        }
        else
        {
            output_extents.erase(output_extents.begin() + static_cast<std::ptrdiff_t>(index));
        }
        return output_extents;
    }

    Shape transpose(const Shape &shape, Index dim0, Index dim1)
    {
        validate_shape(shape);
        const auto shape_rank = rank(shape);
        require_axis_in_range(shape_rank, dim0, "transpose");
        require_axis_in_range(shape_rank, dim1, "transpose");
        auto result = shape;
        std::swap(result[as_size(dim0)], result[as_size(dim1)]);
        return result;
    }

    Coordinates coordinates_from_linear(const Shape &shape, Index linear)
    {
        if (linear < 0 || linear >= numel(shape))
        {
            throw std::out_of_range("linear tensor index is outside the shape");
        }

        Coordinates coordinates(shape.size(), 0);
        for (Index dimension = rank(shape); dimension-- > 0;)
        {
            const auto index = as_size(dimension);
            coordinates[index] = linear % shape[index];
            linear /= shape[index];
        }
        return coordinates;
    }

} // namespace minitensor::detail::shape
