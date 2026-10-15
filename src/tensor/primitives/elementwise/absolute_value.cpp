#include "absolute_value.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
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

	std::vector<std::optional<Tensor>> AbsoluteValuePrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "absolute value VJP expects 1 input" };
		}

		// The derivative of the absolute value function at 0 is defined here as 0,
		// as the sign function returns 0 when the input is 0.
		return { output_cotangent * sign(inputs.front()) };
	}
}
