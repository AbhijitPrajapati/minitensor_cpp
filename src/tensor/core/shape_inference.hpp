#pragma once

namespace minitensor
{
	class Shape;
}

namespace minitensor::detail
{
	// These functions contain centralized logic for operations that return 
	// varying output shapes (e.g. broadcasting, matmul).

	[[nodiscard]] Shape broadcast_shape(const Shape& lhs, const Shape& rhs);
	[[nodiscard]] Shape matmul_output_shape(const Shape& lhs, const Shape& rhs);
}
