#include "negate.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	std::string_view NegatePrimitive::name() const noexcept
	{
		return "negate";
	}

	TensorSpec NegatePrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "negate expects 1 input tensor" };
		}
		if (!is_floating_point(inputs.front().dtype))
		{
			throw std::invalid_argument{ "negate requires a floating-point input" };
		}
		return inputs[0];
	}

}
