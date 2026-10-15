#include "sign.hpp"

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
	std::string_view SignPrimitive::name() const noexcept
	{
		return "sign";
	}

	TensorSpec SignPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "sign expects 1 input tensor" };
		}
		if (!is_floating_point(inputs.front().dtype))
		{
			throw std::invalid_argument{ "sign requires a floating-point input" };
		}
		return inputs.front();
	}

	std::vector<std::optional<Tensor>> SignPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor&) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "sign VJP expects 1 input" };
		}
		// std::nullopt signals no gradient and the end of propagation.
		return { std::nullopt };
	}
}
