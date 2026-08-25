#include "minitensor/ops.hpp"

#include "autograd/node.hpp"
#include "core/shape.hpp"
#include "kernels/kernels.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace minitensor
{
    namespace
    {

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
            const bool needs_grad = lhs.requires_grad() || rhs.requires_grad();
            auto result = detail::make_contiguous_tensor(output_shape, needs_grad);
            detail::kernel::binary(
                detail::kernel::TensorViewAccess::view(lhs),
                detail::kernel::TensorViewAccess::view(rhs),
                detail::kernel::TensorViewAccess::mutable_view(result),
                operation);

            if (needs_grad)
            {
                const auto lhs_shape = lhs.shape();
                const auto rhs_shape = rhs.shape();
                const bool lhs_needs_grad = lhs.requires_grad();
                const bool rhs_needs_grad = rhs.requires_grad();
                const auto saved_lhs = lhs.detach();
                const auto saved_rhs = rhs.detach();

                detail::autograd::set_history(
                    result,
                    name,
                    {lhs, rhs},
                    [operation,
                     lhs_shape,
                     rhs_shape,
                     lhs_needs_grad,
                     rhs_needs_grad,
                     saved_lhs,
                     saved_rhs](const Tensor &gradient)
                    {
                        detail::autograd::GradList gradients(2);
                        switch (operation)
                        {
                        case detail::kernel::BinaryKernel::add:
                            if (lhs_needs_grad)
                            {
                                gradients[0] = detail::reduce_gradient_to_shape(gradient, lhs_shape);
                            }
                            if (rhs_needs_grad)
                            {
                                gradients[1] = detail::reduce_gradient_to_shape(gradient, rhs_shape);
                            }
                            break;
                        case detail::kernel::BinaryKernel::subtract:
                            if (lhs_needs_grad)
                            {
                                gradients[0] = detail::reduce_gradient_to_shape(gradient, lhs_shape);
                            }
                            if (rhs_needs_grad)
                            {
                                gradients[1] = detail::reduce_gradient_to_shape(-gradient, rhs_shape);
                            }
                            break;
                        case detail::kernel::BinaryKernel::multiply:
                            if (lhs_needs_grad)
                            {
                                gradients[0] = detail::reduce_gradient_to_shape(
                                    gradient * saved_rhs, lhs_shape);
                            }
                            if (rhs_needs_grad)
                            {
                                gradients[1] = detail::reduce_gradient_to_shape(
                                    gradient * saved_lhs, rhs_shape);
                            }
                            break;
                        case detail::kernel::BinaryKernel::divide:
                            if (lhs_needs_grad)
                            {
                                gradients[0] = detail::reduce_gradient_to_shape(
                                    gradient / saved_rhs, lhs_shape);
                            }
                            if (rhs_needs_grad)
                            {
                                const auto local_gradient =
                                    -(gradient * saved_lhs) / (saved_rhs * saved_rhs);
                                gradients[1] = detail::reduce_gradient_to_shape(
                                    local_gradient, rhs_shape);
                            }
                            break;
                        }
                        return gradients;
                    });
            }
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
        auto result = detail::make_contiguous_tensor(value.shape(), value.requires_grad());
        detail::kernel::unary(
            detail::kernel::TensorViewAccess::view(value),
            detail::kernel::TensorViewAccess::mutable_view(result),
            detail::kernel::UnaryKernel::negate);
        if (value.requires_grad())
        {
            detail::autograd::set_history(
                result,
                "negate",
                {value},
                [](const Tensor &gradient)
                {
                    return detail::autograd::GradList{
                        std::optional<Tensor>{-gradient}};
                });
        }
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
