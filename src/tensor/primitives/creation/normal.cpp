#include "normal.hpp"

#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/core/random.hpp"
#include "tensor/core/tensor_spec.hpp"


namespace minitensor::detail
{
	NormalParameters::NormalParameters(float mean, float std_dev)
		: mean_(mean), std_dev_(std_dev)
	{
		if (!std::isfinite(mean_) || !std::isfinite(std_dev_))
		{
			throw std::invalid_argument{ "normal parameters must be finite" };
		}
		if (std_dev_ < 0.0F)
		{
			throw std::invalid_argument{ "normal standard deviation cannot be negative" };
		}
	}

	float NormalParameters::mean() const noexcept
	{
		return mean_;
	}

	float NormalParameters::std_dev() const noexcept
	{
		return std_dev_;
	}

	NormalPrimitive::NormalPrimitive(TensorSpec output_spec, NormalParameters parameters, RandomKey key)
		: output_spec_(std::move(output_spec)), parameters_(parameters), key_(key)
	{}

	std::string_view NormalPrimitive::name() const noexcept
	{
		return "normal";
	}

	TensorSpec NormalPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (!inputs.empty())
		{
			throw std::invalid_argument{ "normal expects no input tensors" };
		}
		return output_spec_;
	}

	float NormalPrimitive::mean() const noexcept
	{
		return parameters_.mean();
	}

	float NormalPrimitive::std_dev() const noexcept
	{
		return parameters_.std_dev();
	}

	const RandomKey& NormalPrimitive::key() const noexcept
	{
		return key_;
	}

	std::vector<std::optional<Tensor>> NormalPrimitive::vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor&) const
	{
		if (!inputs.empty())
		{
			throw std::logic_error{ "normal VJP expects no inputs" };
		}
		return {};
	}
}
