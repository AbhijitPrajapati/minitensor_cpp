#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "detail/parameter_updates.hpp"
#include "minitensor/nn/parameters/parameter_id.hpp"

namespace minitensor::persistence
{
class Record;
}

namespace minitensor::nn
{
class AdamW final
{
  public:
    struct StepCoefficients final
    {
        std::uint64_t step;
        float decay;
        float beta1_correction;
        float beta2_correction;
    };
    class State final
    {
      public:
        State(const State &) = default;
        State(State &&) noexcept = default;
        State &operator=(const State &) = default;
        State &operator=(State &&) noexcept = default;

        [[nodiscard]] std::uint64_t step() const noexcept;

      private:
        struct Moment
        {
            Tensor first;
            Tensor second;
        };

        State() = default;

        std::uint64_t step_{0};
        std::unordered_map<ParameterId, Moment> moments_;

        friend class AdamW;
        friend State leafify_optimizer_state(State state);

        template <typename Tree>
        friend persistence::Record capture_optimizer_state(const ParameterSnapshot<Tree> &snapshot, const State &state);

        template <typename Tree>
        friend State restore_adamw_state(const persistence::Record &record, const ParameterSnapshot<Tree> &target);
    };

    AdamW(float learning_rate, float weight_decay, float beta1 = 0.9F, float beta2 = 0.999F, float epsilon = 1e-8F);

    template <typename Tree>
    [[nodiscard]] State initialize(const ParameterSnapshot<Tree> &snapshot) const
    {
        State state;
        for (const auto &[id, parameter] : snapshot.parameters())
        {
            if (!parameter.metadata.trainable)
            {
                continue;
            }

            const Tensor &value = parameter.value;

            if (!is_floating_point(value.dtype()))
            {
                throw std::invalid_argument{"AdamW parameters must be floating point"};
            }

            state.moments_.emplace(id, State::Moment{zeros_like(value), zeros_like(value)});
        }
        return state;
    }

    template <typename Tree>
    [[nodiscard]] std::pair<ParameterSnapshot<Tree>, State> step(ParameterSnapshot<Tree> snapshot, State state,
                                                                 const ParameterGradients &gradients) const
    {
        const StepCoefficients coefficients = step_coefficients(state.step_);

        // Ensure consistency with AdamW state.
        if (gradients.size() != state.moments_.size())
        {
            throw std::invalid_argument{"AdamW state does not match trainable parameters"};
        }

        auto updated = detail::transform_parameter_values_with_gradients(
            std::move(snapshot), gradients, [&](ParameterId id, const Tensor &value, const Tensor &gradient) {
                return update_parameter(id, value, gradient, state, coefficients);
            });

        state.step_ = coefficients.step;
        return {std::move(updated), std::move(state)};
    }

    [[nodiscard]] float learning_rate() const noexcept;
    [[nodiscard]] float weight_decay() const noexcept;
    [[nodiscard]] float beta1() const noexcept;
    [[nodiscard]] float beta2() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;

  private:
    [[nodiscard]] StepCoefficients step_coefficients(std::uint64_t current_step) const;
    [[nodiscard]] Tensor update_parameter(ParameterId id, const Tensor &value, const Tensor &gradient, State &state,
                                          const StepCoefficients &coefficients) const;

    float learning_rate_;
    float weight_decay_;
    float beta1_;
    float beta2_;
    float epsilon_;
};

// Optimizer state must be leafified after each step to release the graph
// retained by its moment tensors.
[[nodiscard]] AdamW::State leafify_optimizer_state(AdamW::State state);

[[nodiscard]] persistence::Record capture_optimizer_configuration(const AdamW &optimizer);
[[nodiscard]] AdamW restore_adamw_configuration(const persistence::Record &record);
} // namespace minitensor::nn
