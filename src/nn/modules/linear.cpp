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
	Linear::InitializedParameters Linear::initialize_parameters(
		Extent input_features,
		Extent output_features,
		RandomKey key,
		bool use_bias,
		TensorOptions options)
	{
		if (input_features <= 0)
		{
			throw std::invalid_argument{ "linear input features must be positive" };
		}
		if (output_features <= 0)
		{
			throw std::invalid_argument{ "linear output features must be positive" };
		}

		auto [weight_key, bias_key] = split(key);

		const float bound = 1.0F / std::sqrt(static_cast<float>(input_features));
		Parameter weight{ uniform(
			Shape{output_features, input_features}, -bound, bound, weight_key, options) };

		std::optional<Parameter> bias;
		if (use_bias)
		{
			bias.emplace(uniform(
				Shape{ output_features }, -bound, bound, bias_key, options));
		}

		return InitializedParameters{ std::move(weight), std::move(bias) };
	}

	Linear::Linear(
		Extent input_features,
		Extent output_features,
		RandomKey key,
		bool use_bias,
		TensorOptions options)
		: Linear{ input_features, output_features, initialize_parameters(input_features, output_features, key, use_bias, options) }
	{}

	Linear::Linear(
		Extent input_features,
		Extent output_features,
		InitializedParameters parameters) noexcept
		: input_features_{ input_features },
		output_features_{ output_features },
		weight_{ std::move(parameters.weight) },
		bias_{ std::move(parameters.bias) }
	{}

	Tensor Linear::operator()(const Tensor& input) const
	{
		Tensor output = matmul(input, transpose(weight_.value(), 0, 1));
		if (bias_)
		{
			output = output + bias_->value();
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

	const Parameter& Linear::weight() const noexcept
	{
		return weight_;
	}

	const Parameter* Linear::bias() const noexcept
	{
		return bias_ ? &*bias_ : nullptr;
	}
}
