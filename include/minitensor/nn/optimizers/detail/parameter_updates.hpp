#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>


namespace minitensor::nn::detail
{
	template <typename Function>
	concept ParameterGradientTransform =
		std::invocable<Function&, const Parameter&, const Tensor&>&&
		std::convertible_to<
		std::invoke_result_t<Function&, const Parameter&, const Tensor&>,
		Tensor>;

	// This function traverses and validates each unique parameter, (skipping frozen parameters)
	// finds the corresponding gradient, and invokes the callback with the validated pair.
	template <typename Tree, ParameterGradientTransform Update>
	[[nodiscard]] Tree transform_parameter_values_with_gradients(
		Tree parameters,
		const ParameterGradients& gradients,
		Update&& update)
	{
		std::size_t update_count = 0;

		Tree updated = transform_parameter_values(
			std::move(parameters),
			[&](const Parameter& parameter)
			{
				if (!parameter.metadata().trainable)
				{
					return parameter.value();
				}

				if (!is_floating_point(parameter.value().dtype()))
				{
					throw std::invalid_argument{ "Optimizer requires floating-point trainable parameters" };
				}

				const ParameterId id = parameter.id();
				const Tensor& value = parameter.value();

				// Find and validate gradient.
				const auto gradient_position = gradients.find(id);
				if (gradient_position == gradients.end())
				{
					throw std::invalid_argument{ "parameter gradient not found" };
				}
				const Tensor& gradient = gradient_position->second;
				if (gradient.shape() != value.shape()
					|| gradient.dtype() != value.dtype()
					|| gradient.device() != value.device())
				{
					throw std::invalid_argument{ "parameter gradient does not match parameter spec" };
				}

				// Perform the update.
				Tensor updated_value = std::invoke(update, parameter, gradient);
				++update_count;
				return updated_value;
			});

		// Ensure that gradients are exhaustive.
		if (update_count != gradients.size())
		{
			throw std::invalid_argument{ "parameter gradients contain extra gradients" };
		}

		return updated;
	}
}
