#include "full.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/host_scalar.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	FullPrimitive::FullPrimitive(TensorSpec output_spec, HostScalar fill_value)
		: output_spec_(std::move(output_spec)),
		fill_value_(fill_value.cast(output_spec_.dtype))
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

	const HostScalar& FullPrimitive::fill_value() const noexcept
	{
		return fill_value_;
	}

}
