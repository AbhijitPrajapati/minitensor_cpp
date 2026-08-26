#include "minitensor/tensor.hpp"

#include "autograd/recording.hpp"
#include "core/layout.hpp"
#include "core/tensor_impl.hpp"
#include "core/shape.hpp"
#include "kernels/kernels.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace minitensor
{
    namespace
    {

        detail::autograd::GradList transpose_backward(
            const Tensor &gradient,
            detail::autograd::TensorSpan,
            detail::autograd::TensorSpan,
            const Index dim0,
            const Index dim1)
        {
            return detail::autograd::GradList{std::optional<Tensor>{
                gradient.transpose(dim0, dim1)}};
        }

        detail::autograd::BackwardFn make_transpose_backward(
            const Index dim0,
            const Index dim1)
        {
            return [dim0, dim1](
                       const Tensor &gradient,
                       const detail::autograd::TensorSpan parents,
                       const detail::autograd::TensorSpan saved_tensors)
            {
                return transpose_backward(
                    gradient, parents, saved_tensors, dim0, dim1);
            };
        }

        detail::autograd::GradList slice_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan,
            const Index dim,
            const Index start,
            const Index stop,
            const Index step)
        {
            auto input_gradient = Tensor::zeros(parents[0].shape());
            const auto input_gradient_view =
                detail::kernel::TensorViewAccess::mutable_view(input_gradient);
            const auto destination_layout =
                input_gradient_view.layout.sliced(dim, start, stop, step);
            detail::kernel::copy(
                detail::kernel::TensorViewAccess::view(gradient),
                detail::kernel::MutableTensorView{
                    input_gradient_view.storage, destination_layout});
            return detail::autograd::GradList{
                std::optional<Tensor>{std::move(input_gradient)}};
        }

        detail::autograd::BackwardFn make_slice_backward(
            const Index dim,
            const Index start,
            const Index stop,
            const Index step)
        {
            return [dim, start, stop, step](
                       const Tensor &gradient,
                       const detail::autograd::TensorSpan parents,
                       const detail::autograd::TensorSpan saved_tensors)
            {
                return slice_backward(
                    gradient, parents, saved_tensors, dim, start, stop, step);
            };
        }

        detail::autograd::GradList contiguous_backward(
            const Tensor &gradient,
            detail::autograd::TensorSpan,
            detail::autograd::TensorSpan)
        {
            return detail::autograd::GradList{
                std::optional<Tensor>{gradient}};
        }

        detail::autograd::GradList reshape_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan)
        {
            return detail::autograd::GradList{std::optional<Tensor>{
                gradient.reshape(parents[0].shape())}};
        }

    } // namespace

    Tensor Tensor::transpose(const Index dim0, const Index dim1) const
    {
        auto transposed_layout = impl_->layout.transposed(dim0, dim1);
        if (dim0 == dim1)
        {
            return *this;
        }

        detail::autograd::OperationContext context{*this};
        auto result = Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage,
            std::move(transposed_layout),
            context.requires_grad()));

        context.record(
            result, "transpose", make_transpose_backward(dim0, dim1));
        return result;
    }

    Tensor Tensor::slice(
        const Index dim,
        const Index start,
        const Index stop,
        const Index step) const
    {
        detail::autograd::OperationContext context{*this};
        auto result = Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage,
            impl_->layout.sliced(dim, start, stop, step),
            context.requires_grad()));

        context.record(
            result, "slice", make_slice_backward(dim, start, stop, step));
        return result;
    }

    Tensor Tensor::contiguous() const
    {
        if (is_contiguous())
        {
            return *this;
        }

        detail::autograd::OperationContext context{*this};
        auto result = Tensor::zeros(shape(), context.requires_grad());
        detail::kernel::copy(
            detail::kernel::TensorViewAccess::view(*this),
            detail::kernel::TensorViewAccess::mutable_view(result));
        context.record(result, "contiguous", contiguous_backward);
        return result;
    }

    Tensor Tensor::reshape(Shape new_shape) const
    {
        detail::shape::require_reshape_compatible(shape(), new_shape);
        if (!is_contiguous())
        {
            return contiguous().reshape(std::move(new_shape));
        }
        if (new_shape == shape())
        {
            return *this;
        }

        detail::autograd::OperationContext context{*this};
        auto result = Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage,
            impl_->layout.reshaped(std::move(new_shape)),
            context.requires_grad()));
        context.record(result, "reshape", reshape_backward);
        return result;
    }

} // namespace minitensor
