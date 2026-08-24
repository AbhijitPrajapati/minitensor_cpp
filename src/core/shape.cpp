#include "core/shape.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace minitensor::detail
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

        std::size_t as_size(const Index value)
        {
            return static_cast<std::size_t>(value);
        }

    } // namespace

    Index shape_rank(const Shape &shape) noexcept
    {
        return static_cast<Index>(shape.size());
    }

    Index shape_numel(const Shape &shape)
    {
        validate_rank(shape.size());
        return checked_numel(shape);
    }

    void require_shape_axis(
        const Shape &shape,
        const Index dimension,
        const std::string_view operation)
    {
        static_cast<void>(shape_numel(shape));
        if (dimension < 0 || dimension >= shape_rank(shape))
        {
            throw std::out_of_range(std::string(operation) + " dimension is out of range");
        }
    }

    void require_reshape_compatible(const Shape &lhs, const Shape &rhs)
    {
        if (!is_reshape_compatible(lhs, rhs))
        {
            throw std::invalid_argument("reshape must preserve the number of elements");
        }
    }

    bool is_reshape_compatible(const Shape &lhs, const Shape &rhs)
    {
        return shape_numel(lhs) == shape_numel(rhs);
    }

    Shape broadcast_shapes(const Shape &lhs, const Shape &rhs)
    {
        static_cast<void>(shape_numel(lhs));
        static_cast<void>(shape_numel(rhs));
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
        static_cast<void>(shape_numel(output_extents));
        return output_extents;
    }

    bool shape_is_broadcastable_to(const Shape &source, const Shape &target)
    {
        static_cast<void>(shape_numel(source));
        static_cast<void>(shape_numel(target));
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

    Shape reduce_shape(const Shape &shape, std::optional<Index> dimension, bool keepdim)
    {
        static_cast<void>(shape_numel(shape));
        if (!dimension.has_value())
        {
            return keepdim ? Shape(shape.size(), 1) : Shape{};
        }

        require_shape_axis(shape, *dimension, "sum");
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

    Shape transpose_shape(const Shape &shape, Index dim0, Index dim1)
    {
        require_shape_axis(shape, dim0, "transpose");
        require_shape_axis(shape, dim1, "transpose");
        auto result = shape;
        std::swap(result[as_size(dim0)], result[as_size(dim1)]);
        return result;
    }

    Shape replace_shape_extent(const Shape &shape, Index dimension, Index extent)
    {
        require_shape_axis(shape, dimension, "shape");
        auto result = shape;
        result[as_size(dimension)] = extent;
        static_cast<void>(shape_numel(result));
        return result;
    }

    Shape insert_shape_axis(const Shape &shape, const Index dimension, const Index extent)
    {
        static_cast<void>(shape_numel(shape));
        if (dimension < 0 || dimension > shape_rank(shape))
        {
            throw std::out_of_range("inserted dimension is out of range");
        }
        auto result = shape;
        result.insert(result.begin() + static_cast<std::ptrdiff_t>(dimension), extent);
        static_cast<void>(shape_numel(result));
        return result;
    }

    Coordinates coordinates_from_linear(const Shape &shape, Index linear)
    {
        const auto numel = shape_numel(shape);
        if (linear < 0 || linear >= numel)
        {
            throw std::out_of_range("linear tensor index is outside the shape");
        }

        Coordinates coordinates(shape.size(), 0);
        for (Index dimension = shape_rank(shape); dimension-- > 0;)
        {
            const auto index = as_size(dimension);
            coordinates[index] = linear % shape[index];
            linear /= shape[index];
        }
        return coordinates;
    }

} // namespace minitensor::detail
