#include <minitensor/nn/modules/linear.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/linalg.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
	Linear::Parameters::Parameters(
		Parameter weight,
		std::optional<Parameter> bias) noexcept
		: weight_{ std::move(weight) }, bias_{ std::move(bias) }
	{}

	Linear::Linear(Extent input_features, Extent output_features, bool use_bias)
		: input_features_{ input_features },
		output_features_{ output_features },
		use_bias_{ use_bias }
	{
		if (input_features_ <= 0)
		{
			throw std::invalid_argument{ "linear input features must be positive" };
		}
		if (output_features_ <= 0)
		{
			throw std::invalid_argument{ "linear output features must be positive" };
		}
	}

	Linear::Parameters Linear::initialize(RandomKey key, TensorOptions options) const
	{
		auto [weight_key, bias_key] = split(key);

		const float bound = 1.0F / std::sqrt(static_cast<float>(input_features_));
		Parameter weight{ uniform(
			Shape{output_features_, input_features_}, -bound, bound, weight_key, options) };

		std::optional<Parameter> bias;
		if (use_bias_)
		{
			bias.emplace(uniform(
				Shape{ output_features_ }, -bound, bound, bias_key, options));
		}

		return Parameters{ std::move(weight), std::move(bias) };
	}

	const Parameter& Linear::Parameters::weight() const noexcept
	{
		return weight_;
	}

	const Parameter* Linear::Parameters::bias() const noexcept
	{
		return bias_ ? &*bias_ : nullptr;
	}

	Tensor Linear::operator()(
		const Parameters& parameters,
		const Tensor& input) const
	{
		Tensor output = matmul(input, transpose(parameters.weight().value(), 0, 1));
		if (const Parameter* bias = parameters.bias())
		{
			output = output + bias->value();
		}
		return output;
	}

	Extent Linear::input_features() const noexcept
	{
		return input_features_;
	}

	Extent Linear::output_features() const noexcept
	{
		return output_features_;
	}

	bool Linear::use_bias() const noexcept
	{
		return use_bias_;
	}
}
