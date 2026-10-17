#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
	class AdamW final
	{
		using MomentMap = std::unordered_map<ParameterId, Tensor>;

		struct StepCoefficients final
		{
			std::uint64_t step;
			float decay;
			float beta1_correction;
			float beta2_correction;
		};

	public:
		class State final
		{
		public:
			State(const State&) = default;
			State(State&&) noexcept = default;
			State& operator=(const State&) = default;
			State& operator=(State&&) noexcept = default;

			[[nodiscard]] std::uint64_t step() const noexcept;

		private:
			State() = default;

			std::uint64_t step_{ 0 };
			MomentMap first_moments_;
			MomentMap second_moments_;

			friend class AdamW;
			friend State leafify_optimizer_state(State state);
		};

		AdamW(
			float learning_rate,
			float weight_decay,
			float beta1 = 0.9F,
			float beta2 = 0.999F,
			float epsilon = 1e-8F);

		template <typename Tree>
		[[nodiscard]] State initialize(const Tree& parameters) const
		{
			State state;
			for_each_unique_parameter(parameters, [&](const Parameter& parameter)
				{
					if (!parameter.metadata().trainable)
					{
						return;
					}

					if (!is_floating_point(parameter.value().dtype()))
					{
						throw std::invalid_argument{ "AdamW requires floating-point trainable parameters" };
					}

					const Tensor& value = parameter.value();
					state.first_moments_.emplace(parameter.id(), zeros_like(value));
					state.second_moments_.emplace(parameter.id(), zeros_like(value));
				});
			return state;
		}

		template <typename Tree>
		[[nodiscard]] std::pair<Tree, State> step(
			Tree parameters,
			State state,
			const ParameterGradients& gradients) const
		{
			const StepCoefficients coefficients = step_coefficients(state.step_);
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
						throw std::invalid_argument{ "AdamW requires floating-point trainable parameters" };
					}

					Tensor updated_value = update_parameter(
						parameter,
						gradients,
						state,
						coefficients);
					++update_count;
					return updated_value;
				});

			if (update_count != gradients.size())
			{
				throw std::invalid_argument{ "parameter gradients contain extra gradients" };
			}
			if (update_count != state.first_moments_.size()
				|| update_count != state.second_moments_.size())
			{
				throw std::invalid_argument{ "AdamW state does not match trainable parameters" };
			}

			state.step_ = coefficients.step;
			return { std::move(updated), std::move(state) };
		}

		[[nodiscard]] float learning_rate() const noexcept;
		[[nodiscard]] float weight_decay() const noexcept;
		[[nodiscard]] float beta1() const noexcept;
		[[nodiscard]] float beta2() const noexcept;
		[[nodiscard]] float epsilon() const noexcept;

	private:
		[[nodiscard]] StepCoefficients step_coefficients(std::uint64_t current_step) const;
		[[nodiscard]] Tensor update_parameter(
			const Parameter& parameter,
			const ParameterGradients& gradients,
			State& state,
			const StepCoefficients& coefficients) const;

		float learning_rate_;
		float weight_decay_;
		float beta1_;
		float beta2_;
		float epsilon_;
	};

	// Optimizer state must be leafified after each step to release the graph
	// retained by its moment tensors.
	[[nodiscard]] AdamW::State leafify_optimizer_state(AdamW::State state);
}
