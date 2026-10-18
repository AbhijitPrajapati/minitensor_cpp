#pragma once

#include <memory>
#include <span>

#include "fwd.hpp"

namespace minitensor::detail
{
class Primitive;
[[nodiscard]] ValueRef apply_primitive(std::unique_ptr<Primitive> primitive, std::span<const ValueRef> inputs);
} // namespace minitensor::detail