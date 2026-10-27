#include <minitensor/nn/optimizers/adamw.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
AdamW::AdamW(float learning_rate, float weight_decay, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), weight_decay_(weight_decay), beta1_(beta1), beta2_(beta2), epsilon_(epsilon)
{
    if (!std::isfinite(learning_rate_) || learning_rate_ < 0.0F)
    {
        throw std::invalid_argument{"AdamW learning rate must be finite and non-negative"};
    }
    if (!std::isfinite(weight_decay_) || weight_decay_ < 0.0F)
    {
        throw std::invalid_argument{"AdamW weight decay must be finite and non-negative"};
    }
    if (!std::isfinite(beta1_) || beta1_ < 0.0F || beta1_ >= 1.0F)
    {
        throw std::invalid_argument{"AdamW beta1 must be in [0, 1)"};
    }
    if (!std::isfinite(beta2_) || beta2_ < 0.0F || beta2_ >= 1.0F)
    {
        throw std::invalid_argument{"AdamW beta2 must be in [0, 1)"};
    }
    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0F)
    {
        throw std::invalid_argument{"AdamW epsilon must be finite and positive"};
    }
    if (!std::isfinite(learning_rate_ * weight_decay_))
    {
        throw std::invalid_argument{"AdamW decay coefficient must be finite"};
    }
}

std::uint64_t AdamW::State::step() const noexcept
{
    return step_;
}

AdamW::StepCoefficients AdamW::step_coefficients(std::uint64_t current_step) const
{
    if (current_step == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error{"AdamW step counter overflow"};
    }

    const std::uint64_t next_step = current_step + 1;
    const double exponent = static_cast<double>(next_step);
    return StepCoefficients{next_step, 1.0F - learning_rate_ * weight_decay_,
                            1.0F - static_cast<float>(std::pow(beta1_, exponent)),
                            1.0F - static_cast<float>(std::pow(beta2_, exponent))};
}

Tensor AdamW::update_parameter(ParameterId id, const Tensor &value, const Tensor &gradient, State &state,
                               const StepCoefficients &coefficients) const
{
    const auto moment_pos = state.moments_.find(id);
    if (moment_pos == state.moments_.end())
    {
        throw std::invalid_argument{"moment not found"};
    }

    State::Moment &moment = moment_pos->second;
    Tensor &first_moment = moment.first;
    Tensor &second_moment = moment.second;
    if (value.shape() != first_moment.shape() || value.dtype() != first_moment.dtype() ||
        value.device() != first_moment.device() || value.shape() != second_moment.shape() ||
        value.dtype() != second_moment.dtype() || value.device() != second_moment.device())
    {
        throw std::invalid_argument{"moment does not match parameter spec"};
    }

    Tensor next_first_moment = beta1_ * first_moment + (1.0F - beta1_) * gradient;
    Tensor next_second_moment = beta2_ * second_moment + (1.0F - beta2_) * gradient * gradient;

    Tensor corrected_first_moment = next_first_moment / coefficients.beta1_correction;
    Tensor corrected_second_moment = next_second_moment / coefficients.beta2_correction;

    first_moment = std::move(next_first_moment);
    second_moment = std::move(next_second_moment);

    return coefficients.decay * value -
           learning_rate_ * corrected_first_moment / (sqrt(corrected_second_moment) + epsilon_);
}

float AdamW::learning_rate() const noexcept
{
    return learning_rate_;
}

float AdamW::weight_decay() const noexcept
{
    return weight_decay_;
}

float AdamW::beta1() const noexcept
{
    return beta1_;
}

float AdamW::beta2() const noexcept
{
    return beta2_;
}

float AdamW::epsilon() const noexcept
{
    return epsilon_;
}

AdamW::State leafify_optimizer_state(AdamW::State state)
{
    // First and second moments are collected in a single pair of vectors.
    std::vector<ParameterId> ids;
    std::vector<Tensor> moments;
    ids.reserve(state.moments_.size());
    moments.reserve(state.moments_.size() * 2);

    for (const auto &[id, moment] : state.moments_)
    {
        ids.push_back(id);
        moments.push_back(moment.first);
        moments.push_back(moment.second);
    }

    std::vector<Tensor> leafified = leafify(moments);

    // First and second moments are split up again after leafification.
    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        state.moments_.at(ids[index]) =
            AdamW::State::Moment{std::move(leafified[2 * index]), std::move(leafified[2 * index + 1])};
    }
    return state;
}
} // namespace minitensor::nn
