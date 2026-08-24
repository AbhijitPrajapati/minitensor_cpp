#pragma once

#include "minitensor/types.hpp"

#include <span>

namespace minitensor::detail
{

    class Layout final
    {
    public:
        // Generates stride from shape given contiguity
        static Layout contiguous(Shape shape, Index offset = 0);

        Layout(Shape shape, Strides strides, Index offset = 0);

        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] const Strides &strides() const noexcept;
        [[nodiscard]] Index offset() const noexcept;
        [[nodiscard]] Index rank() const noexcept;
        [[nodiscard]] Index numel() const noexcept;
        [[nodiscard]] bool is_contiguous() const noexcept;
        [[nodiscard]] Index maximum_offset() const;

        // Converts a logical row-major ordinal to coordinates in this layout's
        // shape. The layout's physical strides do not affect this conversion.
        [[nodiscard]] Coordinates coordinates_from_linear(Index linear) const;

        // Converts validated logical coordinates into an index in the underlying
        // storage, accounting for this layout's strides and base offset.
        [[nodiscard]] Index offset_from_coordinates(std::span<const Index> coordinates) const;

    private:
        Shape shape_;
        Strides strides_;
        Index offset_{0};
        Index numel_{0};
    };

    // Raises error if overflow occurs
    [[nodiscard]] Index checked_numel(const Shape &shape);

} // namespace minitensor::detail
