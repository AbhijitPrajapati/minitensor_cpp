#include "minitensor/ops.hpp"

#include "autograd/node.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <stdexcept>

namespace minitensor {

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rank() != 2 || rhs.rank() != 2) {
        throw std::invalid_argument("matmul requires rank-2 tensors");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("matmul inner dimensions do not match");
    }

    const bool needs_grad = lhs.requires_grad() || rhs.requires_grad();
    auto result = detail::make_contiguous_tensor(
        Shape{lhs.shape()[0], rhs.shape()[1]}, needs_grad);
    detail::matrix_multiply(
        detail::read_arg(lhs), detail::read_arg(rhs), detail::write_arg(result));

    if (needs_grad) {
        const bool lhs_needs_grad = lhs.requires_grad();
        const bool rhs_needs_grad = rhs.requires_grad();
        const auto saved_lhs = lhs.detach();
        const auto saved_rhs = rhs.detach();
        detail::set_history(
            result,
            "matmul",
            {lhs, rhs},
            [lhs_needs_grad, rhs_needs_grad, saved_lhs, saved_rhs](const Tensor& gradient) {
                detail::GradList gradients(2);
                if (lhs_needs_grad) {
                    gradients[0] = matmul(gradient, saved_rhs.transpose(0, 1));
                }
                if (rhs_needs_grad) {
                    gradients[1] = matmul(saved_lhs.transpose(0, 1), gradient);
                }
                return gradients;
            });
    }
    return result;
}

} // namespace minitensor
