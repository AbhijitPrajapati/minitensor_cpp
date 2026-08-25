#pragma once

#include "minitensor/tensor.hpp"

// Shared orchestration helpers for public operations. Numerical work and tensor
// view access remain in the kernel layer, while result construction stays here.
namespace minitensor::detail
{

    [[nodiscard]] Tensor make_contiguous_tensor(
        Shape shape,
        bool requires_grad,
        float fill_value = 0.0F);
    [[nodiscard]] Tensor reduce_gradient_to_shape(
        const Tensor &gradient,
        const Shape &target_shape);
    [[nodiscard]] Tensor slice_gradient(
        const Tensor &gradient,
        const Shape &input_shape,
        Index dim,
        Index start,
        Index stop,
        Index step);
    [[nodiscard]] Tensor relu_gradient(const Tensor &input, const Tensor &gradient);

} // namespace minitensor::detail
