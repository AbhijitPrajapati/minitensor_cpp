#include "contiguous.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	std::string_view ContiguousPrimitive::name() const noexcept
	{
		return "contiguous";
	}

	TensorSpec ContiguousPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "contiguous requires a single input" };
		}
		return inputs.front();
	}

	std::optional<Layout> ContiguousPrimitive::try_derive_shared_layout(
		const TensorSpec& input_spec,
		const Layout& input_layout,
		const TensorSpec& output_spec) const
	{
		if (!input_layout.is_contiguous(input_spec.shape))
		{
			return std::nullopt;
		}
		return Layout::contiguous(output_spec.shape, input_layout.offset());
	}

	std::vector<std::optional<Tensor>> ContiguousPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "contiguous VJP expects 1 input" };
		}
		return { output_cotangent };
	}
}
