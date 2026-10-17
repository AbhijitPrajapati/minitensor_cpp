#include "absolute_value.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	std::string_view AbsoluteValuePrimitive::name() const noexcept
	{
		return "absolute_value";
	}

	TensorSpec AbsoluteValuePrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "absolute value expects 1 input tensor" };
		}
		if (!is_floating_point(inputs.front().dtype))
		{
			throw std::invalid_argument{ "absolute value requires a floating-point input" };
		}
		return inputs.front();
	}

}
