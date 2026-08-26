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

        detail::autograd::GradList add_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            const auto &lhs = parents[0];
            const auto &rhs = parents[1];
            detail::autograd::GradList gradients(2);
            if (lhs.requires_grad())
            {
                gradients[0] = reduce_gradient_to_shape(gradient, lhs.shape());
            }
            if (rhs.requires_grad())
            {
                gradients[1] = reduce_gradient_to_shape(gradient, rhs.shape());
            }
            return gradients;
        }

        detail::autograd::GradList subtract_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            const auto &lhs = parents[0];
            const auto &rhs = parents[1];
            detail::autograd::GradList gradients(2);
            if (lhs.requires_grad())
            {
                gradients[0] = reduce_gradient_to_shape(gradient, lhs.shape());
            }
            if (rhs.requires_grad())
            {
                gradients[1] = reduce_gradient_to_shape(-gradient, rhs.shape());
            }
            return gradients;
        }

        detail::autograd::GradList multiply_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            const auto &lhs = parents[0];
            const auto &rhs = parents[1];
            detail::autograd::GradList gradients(2);
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
            return gradients;
        }

        detail::autograd::GradList divide_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            const auto &lhs = parents[0];
            const auto &rhs = parents[1];
            detail::autograd::GradList gradients(2);
            if (lhs.requires_grad())
            {
                gradients[0] = reduce_gradient_to_shape(
                    gradient / rhs, lhs.shape());
            }
            if (rhs.requires_grad())
            {
                const auto local_gradient = -(gradient * lhs) / (rhs * rhs);
                gradients[1] = reduce_gradient_to_shape(
                    local_gradient, rhs.shape());
            }
            return gradients;
        }

        using BinaryBackward = detail::autograd::GradList (*)(
            const Tensor &,
            detail::autograd::TensorSpan,
            detail::autograd::TensorSpan);

        struct BinaryOperation final
        {
            detail::kernel::BinaryKernel kernel;
            const char *name;
            BinaryBackward backward;
        };

        constexpr BinaryOperation add_operation{
            detail::kernel::BinaryKernel::add, "add", add_backward};
        constexpr BinaryOperation subtract_operation{
            detail::kernel::BinaryKernel::subtract, "subtract", subtract_backward};
        constexpr BinaryOperation multiply_operation{
            detail::kernel::BinaryKernel::multiply, "multiply", multiply_backward};
        constexpr BinaryOperation divide_operation{
            detail::kernel::BinaryKernel::divide, "divide", divide_backward};

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
            const BinaryOperation operation)
        {
            const auto output_shape = detail::shape::broadcast(lhs.shape(), rhs.shape());
            detail::autograd::OperationContext context{lhs, rhs};
            auto result = Tensor::zeros(output_shape, context.requires_grad());
            detail::kernel::binary(
                detail::kernel::TensorViewAccess::view(lhs),
                detail::kernel::TensorViewAccess::view(rhs),
                detail::kernel::TensorViewAccess::mutable_view(result),
                operation.kernel);

            context.record(result, operation.name, operation.backward);
            return result;
        }

    } // namespace

    Tensor operator+(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(lhs, rhs, add_operation);
    }

    Tensor operator-(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(lhs, rhs, subtract_operation);
    }

    Tensor operator*(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(lhs, rhs, multiply_operation);
    }

    Tensor operator/(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_binary_operation(lhs, rhs, divide_operation);
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
