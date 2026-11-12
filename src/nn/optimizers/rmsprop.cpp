#include <minitensor/nn/optimizers/rmsprop.hpp>

#include <cmath>
#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>
#include <minitensor/ops/elementwise.hpp>

namespace minitensor::nn
{
RMSProp::RMSProp(float learning_rate, float alpha, float epsilon)
    : learning_rate_(learning_rate), alpha_(alpha), epsilon_(epsilon)
{
    if (!std::isfinite(learning_rate_) || learning_rate_ < 0.0F)
    {
        throw std::invalid_argument{"RMSProp learning rate must be finite and non-negative"};
    }
    if (!std::isfinite(alpha_) || alpha_ < 0.0F || alpha_ >= 1.0F)
    {
        throw std::invalid_argument{"RMSProp alpha must be in [0, 1)"};
    }
    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0F)
    {
        throw std::invalid_argument{"RMSProp epsilon must be finite and positive"};
    }
}

Tensor RMSProp::update_parameter(ParameterId id, const Tensor &value, const Tensor &gradient, State &state) const
{
    const auto avg_pos = state.moving_average_.find(id);
    if (avg_pos == state.moving_average_.end())
    {
        throw std::invalid_argument{"moving average not found"};
    }
    Tensor &v = avg_pos->second;

    if (value.shape() != v.shape() || value.dtype() != v.dtype() || value.device() != v.device())
    {
        throw std::invalid_argument{"moment does not match parameter spec"};
    }

    Tensor next_v = alpha_ * v + (1 - alpha_) * gradient * gradient;
    v = std::move(next_v);
    // Epsilon is added outside of the square root. This is the uncentered variation of RMSProp.
    return value - learning_rate_ * gradient / (sqrt(v) + epsilon_);
}

float RMSProp::learning_rate() const noexcept
{
    return learning_rate_;
}

float RMSProp::alpha() const noexcept
{
    return alpha_;
}

float RMSProp::epsilon() const noexcept
{
    return epsilon_;
}

RMSProp::State leafify_optimizer_state(RMSProp::State state)
{
    std::vector<ParameterId> ids;
    std::vector<Tensor> moving_averages;
    ids.reserve(state.moving_average_.size());
    moving_averages.reserve(state.moving_average_.size());

    for (const auto &[id, average] : state.moving_average_)
    {
        ids.push_back(id);
        moving_averages.push_back(average);
    }

    std::vector<Tensor> leafified = leafify(moving_averages);

    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        state.moving_average_.at(ids[index]) = std::move(leafified[index]);
    }
    return state;
}
} // namespace minitensor::nn