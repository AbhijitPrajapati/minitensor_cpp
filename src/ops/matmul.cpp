#include "minitensor/ops.hpp"

#include "autograd/recording.hpp"
#include "kernels/kernels.hpp"

#include <stdexcept>

namespace minitensor
{
    namespace
    {

        detail::autograd::GradList matmul_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            detail::autograd::GradList gradients(2);
            if (parents[0].requires_grad())
            {
                gradients[0] = matmul(gradient, parents[1].transpose(0, 1));
            }
            if (parents[1].requires_grad())
            {
                gradients[1] = matmul(parents[0].transpose(0, 1), gradient);
            }
            return gradients;
        }

    } // namespace

    Tensor matmul(const Tensor &lhs, const Tensor &rhs)
    {
        if (lhs.rank() != 2 || rhs.rank() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 tensors");
        }
        if (lhs.shape()[1] != rhs.shape()[0])
        {
            throw std::invalid_argument("matmul inner dimensions do not match");
        }

        detail::autograd::OperationContext context{lhs, rhs};
        auto result = Tensor::zeros(
            Shape{lhs.shape()[0], rhs.shape()[1]}, context.requires_grad());
        detail::kernel::matrix_multiply(
            detail::kernel::TensorViewAccess::view(lhs),
            detail::kernel::TensorViewAccess::view(rhs),
            detail::kernel::TensorViewAccess::mutable_view(result));

        context.record(result, "matmul", matmul_backward);
        return result;
    }

} // namespace minitensor
