#include "square_root.hpp"

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
	std::string_view SquareRootPrimitive::name() const noexcept
	{
		return "sqrt";
	}

	TensorSpec SquareRootPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "sqrt expects 1 input tensor" };
		}
		return inputs.front();
	}

	std::vector<std::optional<Tensor>> SquareRootPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor& output,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "sqrt VJP expects 1 input" };
		}

		const Tensor two = full(
			Shape{},
			2.0F,
			TensorOptions{ output.dtype(), output.device() });
		return { output_cotangent / (two * output) };
	}
}
