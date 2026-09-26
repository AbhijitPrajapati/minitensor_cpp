#include "hyperbolic_tangent.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
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
		return inputs.front();
	}

	std::vector<std::optional<Tensor>> HyperbolicTangentPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor& output,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "tanh VJP expects 1 input" };
		}

		const Tensor one = full(
			Shape{},
			1.0F,
			TensorOptions{ output.dtype(), output.device() });
		return { output_cotangent * (one - output * output) };
	}
}
