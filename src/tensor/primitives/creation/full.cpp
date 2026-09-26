#include "full.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	FullPrimitive::FullPrimitive(TensorSpec output_spec, float fill_value)
		: output_spec_(output_spec), fill_value_(fill_value)
	{}

	std::string_view FullPrimitive::name() const noexcept
	{
		return "full";
	}

	TensorSpec FullPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (!inputs.empty())
		{
			throw std::invalid_argument{ "full expects no input tensors" };
		}
		return output_spec_;
	}

	float FullPrimitive::fill_value() const noexcept
	{
		return fill_value_;
	}

	std::vector<std::optional<Tensor>> FullPrimitive::vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor&) const
	{
		if (!inputs.empty())
		{
			throw std::logic_error{ "full VJP expects no inputs" };
		}
		return {};
	}
}
