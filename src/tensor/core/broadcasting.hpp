#pragma once

#include <span>
#include <vector>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
[[nodiscard]] std::vector<Extent> broadcast_extents(std::span<const Extent> lhs, std::span<const Extent> rhs);

[[nodiscard]] Shape broadcast_shape(const Shape &lhs, const Shape &rhs);
} // namespace minitensor::detail
