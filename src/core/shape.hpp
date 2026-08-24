#pragma once

#include "minitensor/types.hpp"

#include <optional>
#include <string_view>

namespace minitensor::detail
{
    // Shape is a public alias, so private geometry functions validate every
    // shape they accept before deriving new metadata from it.
    [[nodiscard]] Index shape_rank(const Shape &shape) noexcept;
    [[nodiscard]] Index shape_numel(const Shape &shape);

    [[nodiscard]] Shape broadcast_shapes(const Shape &lhs, const Shape &rhs);
    [[nodiscard]] bool shape_is_broadcastable_to(const Shape &source, const Shape &target);
    [[nodiscard]] Shape reduce_shape(const Shape &shape, std::optional<Index> dimension, bool keepdim);
    [[nodiscard]] Shape transpose_shape(const Shape &shape, Index dim0, Index dim1);
    [[nodiscard]] Shape replace_shape_extent(const Shape &shape, Index dimension, Index extent);
    [[nodiscard]] Shape insert_shape_axis(const Shape &shape, Index dimension, Index extent = 1);
    [[nodiscard]] Coordinates coordinates_from_linear(const Shape &shape, Index linear);

    [[nodiscard]] bool is_reshape_compatible(const Shape &lhs, const Shape &rhs);
    void require_reshape_compatible(const Shape &lhs, const Shape &rhs);
    void require_shape_axis(const Shape &shape, Index dimension, std::string_view operation);
} // namespace minitensor::detail
