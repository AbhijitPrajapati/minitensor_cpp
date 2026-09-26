#include "exponential.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	std::string_view ExponentialPrimitive::name() const noexcept
	{
		return "exp";
	}

	TensorSpec ExponentialPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "exp expects 1 input tensor" };
		}
		return inputs.front();
	}

	std::vector<std::optional<Tensor>> ExponentialPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor& output,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "exp VJP expects 1 input" };
		}
		return { output_cotangent * output };
	}
}
