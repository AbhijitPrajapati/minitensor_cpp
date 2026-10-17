#include "hyperbolic_tangent.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	std::string_view HyperbolicTangentPrimitive::name() const noexcept
	{
		return "tanh";
	}

	TensorSpec HyperbolicTangentPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "tanh expects 1 input tensor" };
		}
		if (!is_floating_point(inputs.front().dtype))
		{
			throw std::invalid_argument{ "tanh requires a floating-point input" };
		}
		return inputs.front();
	}

}
