#pragma once

#include <vector>
#include <span>
#include "fwd.hpp"

namespace minitensor::detail
{
    [[nodiscard]] ValueRef apply_operation(PrimitiveRef primitive, std::span<const ValueRef> inputs);
}
