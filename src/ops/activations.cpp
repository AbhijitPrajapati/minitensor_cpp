#include "minitensor/ops.hpp"

#include "autograd/recording.hpp"
#include "kernels/kernels.hpp"

#include <optional>
#include <utility>

namespace minitensor
{
    namespace
    {

        detail::autograd::GradList relu_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            auto input_gradient = Tensor::zeros(parents[0].shape());
            detail::kernel::relu_backward(
                detail::kernel::TensorViewAccess::view(parents[0]),
                detail::kernel::TensorViewAccess::view(gradient),
                detail::kernel::TensorViewAccess::mutable_view(input_gradient));
            return detail::autograd::GradList{
                std::optional<Tensor>{std::move(input_gradient)}};
        }

        detail::autograd::GradList sigmoid_backward(
            const Tensor &gradient,
            detail::autograd::TensorSpan,
            const detail::autograd::TensorSpan saved_tensors)
        {
            const auto &output = saved_tensors[0];
            auto input_gradient = gradient * output * (1.0F - output);
            return detail::autograd::GradList{
                std::optional<Tensor>{std::move(input_gradient)}};
        }

        detail::autograd::GradList tanh_backward(
            const Tensor &gradient,
            detail::autograd::TensorSpan,
            const detail::autograd::TensorSpan saved_tensors)
        {
            const auto &output = saved_tensors[0];
            auto input_gradient = gradient * (1.0F - output * output);
            return detail::autograd::GradList{
                std::optional<Tensor>{std::move(input_gradient)}};
        }

    } // namespace

    Tensor relu(const Tensor &input)
    {
        detail::autograd::OperationContext context{input};
        auto result = Tensor::zeros(input.shape(), context.requires_grad());
        detail::kernel::unary(
            detail::kernel::TensorViewAccess::view(input),
            detail::kernel::TensorViewAccess::mutable_view(result),
            detail::kernel::UnaryKernel::relu);
        context.record(result, "relu", relu_backward);
        return result;
    }

    Tensor sigmoid(const Tensor &input)
    {
        detail::autograd::OperationContext context{input};
        auto result = Tensor::zeros(input.shape(), context.requires_grad());
        detail::kernel::unary(
            detail::kernel::TensorViewAccess::view(input),
            detail::kernel::TensorViewAccess::mutable_view(result),
            detail::kernel::UnaryKernel::sigmoid);
        context.record(result, "sigmoid", sigmoid_backward, {result});
        return result;
    }

    Tensor tanh(const Tensor &input)
    {
        detail::autograd::OperationContext context{input};
        auto result = Tensor::zeros(input.shape(), context.requires_grad());
        detail::kernel::unary(
            detail::kernel::TensorViewAccess::view(input),
            detail::kernel::TensorViewAccess::mutable_view(result),
            detail::kernel::UnaryKernel::tanh);
        context.record(result, "tanh", tanh_backward, {result});
        return result;
    }

} // namespace minitensor
