#pragma once

#include <minitensor/types.hpp>

namespace minitensor::detail
{
    [[nodiscard]] Shape broadcast_shape(const Shape &lhs, const Shape &rhs);
}