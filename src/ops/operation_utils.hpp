#pragma once

#include "kernels/kernels.hpp"
#include "minitensor/tensor.hpp"

// Shared orchestration helpers for public operations. Numerical work remains in
// kernels, while tensor construction remains behind the private access bridge.
namespace minitensor::detail
{

    [[nodiscard]] ReadTensorArg read_arg(const Tensor &tensor) noexcept;
    [[nodiscard]] WriteTensorArg write_arg(Tensor &tensor) noexcept;
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
        Index step);
    [[nodiscard]] Tensor relu_gradient(const Tensor &input, const Tensor &gradient);

} // namespace minitensor::detail
