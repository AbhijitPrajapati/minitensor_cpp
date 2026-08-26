#pragma once

#include "kernels/tensor_view.hpp"

#include <optional>

namespace minitensor::detail::kernel
{

    enum class BinaryKernel
    {
        add,
        subtract,
        multiply,
        divide
    };
    enum class UnaryKernel
    {
        negate,
        relu,
        sigmoid,
        tanh
    };

    void fill(MutableTensorView output, float value);
    void copy(ConstTensorView input, MutableTensorView output);
    void binary(
        ConstTensorView lhs,
        ConstTensorView rhs,
        MutableTensorView output,
        BinaryKernel operation);
    void unary(ConstTensorView input, MutableTensorView output, UnaryKernel operation);
    void matrix_multiply(
        ConstTensorView lhs,
        ConstTensorView rhs,
        MutableTensorView output);
    void reduce_sum(
        ConstTensorView input,
        MutableTensorView output,
        std::optional<Index> dim,
        bool keepdim);
    void sum_to_shape(ConstTensorView input, MutableTensorView output);
    void relu_backward(
        ConstTensorView input,
        ConstTensorView gradient,
        MutableTensorView output);

} // namespace minitensor::detail::kernel
