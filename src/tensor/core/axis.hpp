#pragma once

#include <minitensor/types.hpp>

namespace minitensor::detail
{
	[[nodiscard]] Shape::size_type normalize_axis(Axis axis, Shape::size_type rank);
}