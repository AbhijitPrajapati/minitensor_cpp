#include <minitensor/nn/loss.hpp>

#include <stdexcept>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
Tensor mse_loss(const Tensor &prediction, const Tensor &target)
{
    if (prediction.shape() != target.shape())
    {
        throw std::invalid_argument{"mse loss requires prediction and target to have the same shape"};
    }
    Tensor error = prediction - target;
    return mean(error * error);
}

Tensor l1_loss(const Tensor &prediction, const Tensor &target)
{
    if (prediction.shape() != target.shape())
    {
        throw std::invalid_argument{"l1 loss requires prediction and target to have the same shape"};
    }
    return mean(abs(prediction - target));
}

Tensor binary_cross_entropy_with_logits(const Tensor &logits, const Tensor &target)
{
    if (logits.shape() != target.shape())
    {
        throw std::invalid_argument{"binary cross entropy requires logits and target to have the same shape"};
    }
    return mean(logaddexp(0.0F, logits) - target * logits);
}

Tensor categorical_cross_entropy_with_logits(const Tensor &logits, const Tensor &targets, Axis class_axis)
{
    if (logits.rank() == 0 || targets.rank() != logits.rank() - 1)
    {
        throw std::invalid_argument{"categorical cross entropy targets must have one fewer dimension than logits"};
    }

    const Tensor expanded_targets = unsqueeze(targets, class_axis);
    const Tensor normalizer = logsumexp(logits, {class_axis}, true);
    if (expanded_targets.shape() != normalizer.shape())
    {
        throw std::invalid_argument{"categorical cross entropy target shape does not match logits"};
    }

    const Tensor selected_logits = gather(logits, class_axis, expanded_targets);
    return mean(normalizer - selected_logits);
}
} // namespace minitensor::nn
