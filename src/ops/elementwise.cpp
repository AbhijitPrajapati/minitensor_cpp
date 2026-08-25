#include "minitensor/ops.hpp"

#include "autograd/recording.hpp"
#include "core/shape.hpp"
#include "kernels/kernels.hpp"

#include <optional>
#include <vector>

namespace minitensor
{
    namespace
    {

        Tensor reduce_gradient_to_shape(
            const Tensor &gradient,
            const Shape &target_shape)
        {
            if (gradient.shape() == target_shape)
            {
                return gradient;
            }

            auto result = Tensor::zeros(target_shape);
            detail::kernel::sum_to_shape(
                detail::kernel::TensorViewAccess::view(gradient),
                detail::kernel::TensorViewAccess::mutable_view(result));
            return result;
        }

        detail::autograd::GradList binary_backward(
            const detail::kernel::BinaryKernel operation,
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents)
        {
            const auto &lhs = parents[0];
            const auto &rhs = parents[1];
            detail::autograd::GradList gradients(2);

            switch (operation)
            {
            case detail::kernel::BinaryKernel::add:
                if (lhs.requires_grad())
                {
                    gradients[0] = reduce_gradient_to_shape(gradient, lhs.shape());
                }
                if (rhs.requires_grad())
                {
                    gradients[1] = reduce_gradient_to_shape(gradient, rhs.shape());
                }
                break;
            case detail::kernel::BinaryKernel::subtract:
                if (lhs.requires_grad())
                {
                    gradients[0] = reduce_gradient_to_shape(gradient, lhs.shape());
                }
                if (rhs.requires_grad())
                {
                    gradients[1] = reduce_gradient_to_shape(-gradient, rhs.shape());
                }
                break;
            case detail::kernel::BinaryKernel::multiply:
                if (lhs.requires_grad())
                {
                    gradients[0] = reduce_gradient_to_shape(
                        gradient * rhs, lhs.shape());
                }
                if (rhs.requires_grad())
                {
                    gradients[1] = reduce_gradient_to_shape(
                        gradient * lhs, rhs.shape());
                }
                break;
            case detail::kernel::BinaryKernel::divide:
                if (lhs.requires_grad())
                {
                    gradients[0] = reduce_gradient_to_shape(
                        gradient / rhs, lhs.shape());
                }
                if (rhs.requires_grad())
                {
                    const auto local_gradient =
                        -(gradient * lhs) / (rhs * rhs);
                    gradients[1] = reduce_gradient_to_shape(
                        local_gradient, rhs.shape());
                }
                break;
            }
            return gradients;
        }

        detail::autograd::BackwardFn make_binary_backward(
            const detail::kernel::BinaryKernel operation)
        {
            return [operation](
                       const Tensor &gradient,
                       const detail::autograd::TensorSpan parents,
                       detail::autograd::TensorSpan)
            { return binary_backward(operation, gradient, parents); };
        }

        detail::autograd::GradList negate_backward(
            const Tensor &gradient,
            detail::autograd::TensorSpan,
            detail::autograd::TensorSpan)
        {
            return detail::autograd::GradList{
                std::optional<Tensor>{-gradient}};
        }

        Tensor scalar_tensor(const float value)
        {
            return Tensor::from_data(std::vector<float>{value}, Shape{});
        }

        Tensor apply_binary_operation(
            const Tensor &lhs,
            const Tensor &rhs,
            const detail::kernel::BinaryKernel operation,
            const char *name)
        {
            const auto output_shape = detail::shape::broadcast(lhs.shape(), rhs.shape());
            detail::autograd::OperationContext context{lhs, rhs};
            auto result = Tensor::zeros(output_shape, context.requires_grad());
            detail::kernel::binary(
                detail::kernel::TensorViewAccess::view(lhs),
                detail::kernel::TensorViewAccess::view(rhs),
                detail::kernel::TensorViewAccess::mutable_view(result),
                operation);

            context.record(result, name, make_binary_backward(operation));
            return result;
        }

    } // namespace

    Tensor operator+(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(lhs, rhs, detail::kernel::BinaryKernel::add, "add");
    }

    Tensor operator-(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(
            lhs, rhs, detail::kernel::BinaryKernel::subtract, "subtract");
    }

    Tensor operator*(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(
            lhs, rhs, detail::kernel::BinaryKernel::multiply, "multiply");
    }

    Tensor operator/(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(lhs, rhs, detail::kernel::BinaryKernel::divide, "divide");
    }

    Tensor operator-(const Tensor &value)
    {
        detail::autograd::OperationContext context{value};
        auto result = Tensor::zeros(value.shape(), context.requires_grad());
        detail::kernel::unary(
            detail::kernel::TensorViewAccess::view(value),
            detail::kernel::TensorViewAccess::mutable_view(result),
            detail::kernel::UnaryKernel::negate);
        context.record(result, "negate", negate_backward);
        return result;
    }

    Tensor operator+(const Tensor &tensor, const float scalar) { return tensor + scalar_tensor(scalar); }
    Tensor operator+(const float scalar, const Tensor &tensor) { return scalar_tensor(scalar) + tensor; }
    Tensor operator-(const Tensor &tensor, const float scalar) { return tensor - scalar_tensor(scalar); }
    Tensor operator-(const float scalar, const Tensor &tensor) { return scalar_tensor(scalar) - tensor; }
    Tensor operator*(const Tensor &tensor, const float scalar) { return tensor * scalar_tensor(scalar); }
    Tensor operator*(const float scalar, const Tensor &tensor) { return scalar_tensor(scalar) * tensor; }
    Tensor operator/(const Tensor &tensor, const float scalar) { return tensor / scalar_tensor(scalar); }
    Tensor operator/(const float scalar, const Tensor &tensor) { return scalar_tensor(scalar) / tensor; }

} // namespace minitensor
