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

        // View transformations derive a new validated layout without changing
        // storage ownership or materializing tensor data.
        [[nodiscard]] Layout transposed(Index dim0, Index dim1) const;
        [[nodiscard]] Layout sliced(Index dim, Index start, Index stop, Index step = 1) const;
        [[nodiscard]] Layout reshaped(Shape shape) const;
        [[nodiscard]] Layout broadcast_to(const Shape &shape) const;

        // Converts validated logical coordinates into an index in the underlying
        // storage, accounting for this layout's strides and base offset.
        [[nodiscard]] Index offset_from_coordinates(std::span<const Index> coordinates) const;

    private:
        Shape shape_;
        Strides strides_;
        Index offset_{0};
        Index numel_{0};
    };

} // namespace minitensor::detail
