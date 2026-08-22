#include "minitensor/ops.hpp"

#include "autograd/node.hpp"
#include "ops/operation_utils.hpp"

#include <optional>

namespace minitensor {

Tensor relu(const Tensor& input) {
    auto result = detail::make_contiguous_tensor(input.shape(), input.requires_grad());
    detail::unary(
        detail::read_arg(input),
        detail::write_arg(result),
        detail::UnaryKernel::relu);
    if (input.requires_grad()) {
        const auto saved_input = input.detach();
        detail::set_history(
            result,
            "relu",
            {input},
            [saved_input](const Tensor& gradient) {
                return detail::GradList{std::optional<Tensor>{
                    detail::relu_gradient(saved_input, gradient)}};
            });
    }
    return result;
}

Tensor sigmoid(const Tensor& input) {
    auto result = detail::make_contiguous_tensor(input.shape(), input.requires_grad());
    detail::unary(
        detail::read_arg(input),
        detail::write_arg(result),
        detail::UnaryKernel::sigmoid);
    if (input.requires_grad()) {
        const auto saved_output = result.detach();
        detail::set_history(
            result,
            "sigmoid",
            {input},
            [saved_output](const Tensor& gradient) {
                auto input_gradient = gradient * saved_output * (1.0F - saved_output);
                return detail::GradList{
                    std::optional<Tensor>{std::move(input_gradient)}};
            });
    }
    return result;
}

Tensor tanh(const Tensor& input) {
    auto result = detail::make_contiguous_tensor(input.shape(), input.requires_grad());
    detail::unary(
        detail::read_arg(input),
        detail::write_arg(result),
        detail::UnaryKernel::tanh);
    if (input.requires_grad()) {
        const auto saved_output = result.detach();
        detail::set_history(
            result,
            "tanh",
            {input},
            [saved_output](const Tensor& gradient) {
                auto input_gradient = gradient * (1.0F - saved_output * saved_output);
                return detail::GradList{
                    std::optional<Tensor>{std::move(input_gradient)}};
            });
    }
    return result;
}

} // namespace minitensor
