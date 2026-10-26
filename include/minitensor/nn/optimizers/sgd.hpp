#pragma once

#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>

#include "detail/parameter_updates.hpp"

namespace minitensor::nn
{
class SGD final
{
  public:
    // SGD has no state.
    class State final
    {
    };

    explicit SGD(float learning_rate);

    template <typename Tree>
    [[nodiscard]] State initialize(const ParameterSnapshot<Tree> &) const noexcept
    {
        return {};
    }

    template <typename Tree>
    [[nodiscard]] std::pair<ParameterSnapshot<Tree>, State> step(const ParameterSnapshot<Tree> &snapshot, State state,
                                                                 const ParameterGradients &gradients) const
    {
        auto updated = detail::transform_parameter_values_with_gradients(
            snapshot, gradients, [&](ParameterId, const Tensor &parameter, const Tensor &gradient) {
                return parameter - learning_rate_ * gradient;
            });
        return {std::move(updated), std::move(state)};
    }

    [[nodiscard]] float learning_rate() const noexcept;

  private:
    float learning_rate_;
};

// SGD has no tensor-valued state, so leafification is a no-op.
[[nodiscard]] SGD::State leafify_optimizer_state(SGD::State state) noexcept;
} // namespace minitensor::nn
