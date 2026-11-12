#pragma once

#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>

#include "minitensor/nn/optimizers/detail/parameter_updates.hpp"

namespace minitensor::nn
{
class RMSProp final
{
  public:
    class State final
    {
      public:
        State(const State &) = default;
        State(State &&) noexcept = default;
        State &operator=(const State &) = default;
        State &operator=(State &&) noexcept = default;

      private:
        State() = default;

        std::unordered_map<ParameterId, Tensor> moving_average_;

        friend class RMSProp;
        friend State leafify_optimizer_state(State state);
    };

    RMSProp(float learning_rate, float alpha = 0.9F, float epsilon = 1e-8F);

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
                throw std::invalid_argument{"RMSProp parameters must be floating point"};
            }
            state.moving_average_.emplace(id, zeros_like(value));
        }
        return state;
    }

    template <typename Tree>
    [[nodiscard]] std::pair<ParameterSnapshot<Tree>, State> step(ParameterSnapshot<Tree> snapshot, State state,
                                                                 const ParameterGradients &gradients) const
    {
        // Ensure consistency with RMSProp state.
        if (gradients.size() != state.moving_average_.size())
        {
            throw std::invalid_argument{"RMSProp state does not match trainable parameters"};
        }
        auto updated = detail::transform_parameter_values_with_gradients(
            std::move(snapshot), gradients, [&](ParameterId id, const Tensor &value, const Tensor &gradient) {
                return update_parameter(id, value, gradient, state);
            });
        return {std::move(updated), std::move(state)};
    }

    [[nodiscard]] float learning_rate() const noexcept;
    [[nodiscard]] float alpha() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;

  private:
    [[nodiscard]] Tensor update_parameter(ParameterId id, const Tensor &value, const Tensor &gradient,
                                          State &state) const;

    float learning_rate_;
    float alpha_;
    float epsilon_;
};

[[nodiscard]] RMSProp::State leafify_optimizer_state(RMSProp::State state);
} // namespace minitensor::nn