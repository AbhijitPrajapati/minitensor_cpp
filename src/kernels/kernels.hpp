#pragma once

#include "core/tensor_access.hpp"

#include <optional>
#include <span>

namespace minitensor::detail::kernel
{

    using ReadTensorArg = ConstTensorView;
    using WriteTensorArg = MutableTensorView;

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

    void fill(WriteTensorArg output, float value);
    void copy(ReadTensorArg input, WriteTensorArg output);
    void binary(
        ReadTensorArg lhs,
        ReadTensorArg rhs,
        WriteTensorArg output,
        BinaryKernel operation);
    void unary(ReadTensorArg input, WriteTensorArg output, UnaryKernel operation);
    void matrix_multiply(ReadTensorArg lhs, ReadTensorArg rhs, WriteTensorArg output);
    void reduce_sum(
        ReadTensorArg input,
        WriteTensorArg output,
        std::optional<Index> dim,
        bool keepdim);
    void sum_to_shape(ReadTensorArg input, WriteTensorArg output);
    void relu_backward(ReadTensorArg input, ReadTensorArg gradient, WriteTensorArg output);

} // namespace minitensor::detail::kernel
