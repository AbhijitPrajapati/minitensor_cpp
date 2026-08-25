#include "ops/operation_utils.hpp"

#include "kernels/kernels.hpp"
#include "kernels/tensor_view.hpp"

#include <utility>

namespace minitensor::detail
{

    Tensor make_contiguous_tensor(
        Shape shape,
        const bool requires_grad,
        const float fill_value)
    {
        auto result = Tensor::zeros(std::move(shape), requires_grad);
        if (fill_value != 0.0F)
        {
            kernel::fill(kernel::TensorViewAccess::mutable_view(result), fill_value);
        }
        return result;
    }

    Tensor reduce_gradient_to_shape(const Tensor &gradient, const Shape &target_shape)
    {
        if (gradient.shape() == target_shape)
        {
            return gradient.detach();
        }
        auto result = make_contiguous_tensor(target_shape, false);
        kernel::sum_to_shape(
            kernel::TensorViewAccess::view(gradient),
            kernel::TensorViewAccess::mutable_view(result));
        return result;
    }

    Tensor slice_gradient(
        const Tensor &gradient,
        const Shape &input_shape,
        const Index dim,
        const Index start,
        const Index stop,
        const Index step)
    {
        auto result = make_contiguous_tensor(input_shape, false);
        const auto result_view = kernel::TensorViewAccess::mutable_view(result);
        const auto destination_layout = result_view.layout.sliced(dim, start, stop, step);
        kernel::copy(
            kernel::TensorViewAccess::view(gradient),
            kernel::MutableTensorView{result_view.storage, destination_layout});
        return result;
    }

    Tensor relu_gradient(const Tensor &input, const Tensor &gradient)
    {
        auto result = make_contiguous_tensor(input.shape(), false);
        kernel::relu_backward(
            kernel::TensorViewAccess::view(input),
            kernel::TensorViewAccess::view(gradient),
            kernel::TensorViewAccess::mutable_view(result));
        return result;
    }

} // namespace minitensor::detail
