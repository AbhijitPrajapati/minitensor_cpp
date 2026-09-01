#pragma once

#include <span>

#include "fwd.hpp"

namespace minitensor::detail
{
    class Primitive;
    [[nodiscard]] ValueRef apply_operation(std::unique_ptr<Primitive> primitive, std::span<const ValueRef> inputs);
}
