#include "minitensor/tensor.hpp"

#include "autograd/node.hpp"
#include "core/layout.hpp"
#include "core/tensor_impl.hpp"
#include "core/shape.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <memory>
#include <utility>

namespace minitensor
{

    Tensor Tensor::transpose(const Index dim0, const Index dim1) const
    {
        auto transposed_layout = impl_->layout.transposed(dim0, dim1);
        if (dim0 == dim1)
        {
            return *this;
        }

        auto result = Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage,
            std::move(transposed_layout),
            requires_grad()));

        if (requires_grad())
        {
            detail::set_history(
                result,
                "transpose",
                {*this},
                [dim0, dim1](const Tensor &gradient)
                {
                    return detail::GradList{
                        std::optional<Tensor>{gradient.transpose(dim0, dim1)}};
                });
        }
        return result;
    }

    Tensor Tensor::slice(
        const Index dim,
        const Index start,
        const Index stop,
        const Index step) const
    {
        const auto input_shape = shape();
        auto result = Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage,
            impl_->layout.sliced(dim, start, stop, step),
            requires_grad()));

        if (requires_grad())
        {
            detail::set_history(
                result,
                "slice",
                {*this},
                [input_shape, dim, start, stop, step](const Tensor &gradient)
                {
                    return detail::GradList{std::optional<Tensor>{
                        detail::slice_gradient(gradient, input_shape, dim, start, stop, step)}};
                });
        }
        return result;
    }

    Tensor Tensor::contiguous() const
    {
        if (is_contiguous())
        {
            return *this;
        }

        auto result = detail::make_contiguous_tensor(shape(), requires_grad());
        detail::copy(detail::read_arg(*this), detail::write_arg(result));
        if (requires_grad())
        {
            detail::set_history(
                result,
                "contiguous",
                {*this},
                [](const Tensor &gradient)
                {
                    return detail::GradList{std::optional<Tensor>{gradient.detach()}};
                });
        }
        return result;
    }

    Tensor Tensor::reshape(Shape new_shape) const
    {
        detail::require_reshape_compatible(shape(), new_shape);
        if (!is_contiguous())
        {
            return contiguous().reshape(std::move(new_shape));
        }
        if (new_shape == shape())
        {
            return *this;
        }

        const auto input_shape = shape();
        auto result = Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage,
            impl_->layout.reshaped(std::move(new_shape)),
            requires_grad()));
        if (requires_grad())
        {
            detail::set_history(
                result,
                "reshape",
                {*this},
                [input_shape](const Tensor &gradient)
                {
                    return detail::GradList{
                        std::optional<Tensor>{gradient.reshape(input_shape)}};
                });
        }
        return result;
    }

} // namespace minitensor
