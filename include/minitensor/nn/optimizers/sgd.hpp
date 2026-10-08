#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	class SGD final
	{
	public:
		struct State final
		{};

		explicit SGD(float learning_rate);

		template <typename Tree>
		[[nodiscard]] State initialize(const Tree&) const noexcept
		{
			return {};
		}

		template <typename Tree>
		[[nodiscard]] std::pair<Tree, State> step(
			Tree parameters,
			State state,
			const ParameterGradients& gradients) const
		{
			std::size_t update_count = 0;

			Tree updated = transform_parameter_values(
				std::move(parameters),
				[&](const Parameter& parameter)
				{
					const Tensor& value = parameter.value();
					if (!parameter.metadata().trainable)
					{
						return value;
					}
					const auto gradient_position = gradients.find(parameter.id());
					if (gradient_position == gradients.end())
					{
						throw std::invalid_argument{ "parameter gradient not found" };
					}
					const Tensor& gradient = gradient_position->second;
					if (gradient.device() != value.device()
						|| gradient.dtype() != value.dtype()
						|| gradient.shape() != value.shape())
					{
						throw std::invalid_argument{ "parameter gradient does not match parameter spec" };
					}
					++update_count;
					return value - learning_rate_ * gradient;
				});
			if (update_count != gradients.size())
			{
				throw std::invalid_argument{ "parameter gradients contain extra gradients" };
			}

			return { std::move(updated), std::move(state) };
		}

		[[nodiscard]] float learning_rate() const noexcept;

	private:
		float learning_rate_;
	};
}
