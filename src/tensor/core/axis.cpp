#include "axis.hpp"

#include <limits>
#include <stdexcept>

namespace minitensor::detail
{
	Shape::size_type normalize_axis(Axis axis, Shape::size_type rank)
	{
		constexpr auto max_axis = std::numeric_limits<Axis>::max();
		if (rank > static_cast<Shape::size_type>(max_axis))
		{
			throw std::overflow_error{"rank exceeds maximum axis"};
		}

		const auto signed_rank = static_cast<Axis>(rank);
		if (axis < -signed_rank || axis >= signed_rank)
		{
			throw std::out_of_range{"axis is out of range for rank"};
		}

		if (axis < 0)
		{
			axis += signed_rank;
		}
		return static_cast<Shape::size_type>(axis);
	}
}