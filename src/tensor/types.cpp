#include <minitensor/types.hpp>
#include <span>
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace minitensor
{
    Shape::Shape(std::initializer_list<value_type> dimensions)
        : dimensions_{dimensions}, numel_(validate_and_count(dimensions)) {}

    Shape::Shape(std::vector<value_type> dimensions)
        : dimensions_{std::move(dimensions)}, numel_(validate_and_count(dimensions)) {}

    Shape::size_type Shape::rank() const noexcept
    {
        return dimensions_.size();
    }

    bool Shape::is_scalar() const noexcept
    {
        return dimensions_.empty();
    }

    Shape::value_type Shape::operator[](size_type axis) const noexcept
    {
        return dimensions_[axis];
    }

    std::span<const Shape::value_type> Shape::dimensions() const noexcept
    {
        return dimensions_;
    }

    Shape::size_type Shape::validate_and_count(std::span<const value_type> dimensions)
    {
        bool contains_zero = false;
        for (const value_type extent : dimensions)
        {
            if (extent < 0)
            {
                throw std::invalid_argument{"tensor extents cannot be negative"};
            }
            contains_zero |= extent == 0;
        }

        // The entire shape has zero elements regardless of the other
        // extents, so don't report multiplication overflow
        if (contains_zero)
        {
            return 0;
        }

        size_type count = 1;

        for (const value_type extent : dimensions)
        {
            const auto converted = static_cast<size_type>(extent);
            if (count > std::numeric_limits<size_type>::max() / converted)
            {
                throw std::overflow_error{"tensor element count overflow"};
            }
            count *= converted;
        }

        return count;
    }

    std::size_t Shape::numel() const noexcept
    {
        return numel_;
    }

    std::size_t dtype_size(DType dtype)
    {
        switch (dtype)
        {
        case DType::Float32:
            return sizeof(float);
        }
        throw std::invalid_argument{"unrecognized dtype"};
    }

    std::string_view dtype_name(DType dtype)
    {
        switch (dtype)
        {
        case DType::Float32:
            return "float32";
        }
        throw std::invalid_argument{"unrecognized dtype"};
    }
}