#pragma once

#include "minitensor/types.hpp"

#include <optional>
#include <string_view>

namespace minitensor::detail::shape
{
    // Shape is a public alias, so private geometry functions validate every
    // shape they accept before deriving new metadata from it.
    [[nodiscard]] Index rank(const Shape &shape) noexcept;
    [[nodiscard]] Index numel(const Shape &shape);

    [[nodiscard]] Shape broadcast(const Shape &lhs, const Shape &rhs);
    [[nodiscard]] bool is_broadcastable_to(const Shape &source, const Shape &target);
    [[nodiscard]] Shape reduce(const Shape &shape, std::optional<Index> dimension, bool keepdim);
    [[nodiscard]] Shape transpose(const Shape &shape, Index dim0, Index dim1);
    [[nodiscard]] Shape replace_extent(const Shape &shape, Index dimension, Index extent);
    [[nodiscard]] Shape insert_axis(const Shape &shape, Index dimension, Index extent = 1);
    [[nodiscard]] Coordinates coordinates_from_linear(const Shape &shape, Index linear);

    [[nodiscard]] bool is_reshape_compatible(const Shape &lhs, const Shape &rhs);
    void require_reshape_compatible(const Shape &lhs, const Shape &rhs);
    void require_axis(const Shape &shape, Index dimension, std::string_view operation);
} // namespace minitensor::detail::shape
