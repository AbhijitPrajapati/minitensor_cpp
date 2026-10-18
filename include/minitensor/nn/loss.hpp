#pragma once

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
[[nodiscard]] Tensor mse_loss(const Tensor &prediction, const Tensor &target);
[[nodiscard]] Tensor l1_loss(const Tensor &prediction, const Tensor &target);
[[nodiscard]] Tensor binary_cross_entropy_with_logits(const Tensor &logits, const Tensor &target);
[[nodiscard]] Tensor categorical_cross_entropy_with_logits(const Tensor &logits, const Tensor &targets,
                                                           Axis class_axis = -1);
} // namespace minitensor::nn
