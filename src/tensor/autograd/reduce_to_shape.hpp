#pragma once

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::detail
{
	[[nodiscard]] Tensor reduce_to_shape(const Tensor& cotangent, const Shape& target_shape);
}