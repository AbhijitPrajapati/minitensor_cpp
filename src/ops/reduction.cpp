#include "minitensor/ops.hpp"

#include "autograd/recording.hpp"
#include "core/shape.hpp"
#include "kernels/kernels.hpp"

#include <cstddef>
#include <optional>
#include <utility>

namespace minitensor
{
    namespace
    {

        detail::autograd::GradList sum_backward(
            const Tensor &gradient,
            const detail::autograd::TensorSpan parents,
            detail::autograd::TensorSpan,
            const std::optional<Index> dim,
            const bool keepdim)
        {
            auto broadcastable = gradient;
            if (dim.has_value() && !keepdim)
            {
                // Restore the singleton axis removed by the forward reduction.
                auto expanded_shape = gradient.shape();
                expanded_shape.insert(
                    expanded_shape.begin() + static_cast<std::ptrdiff_t>(*dim),
                    1);
                broadcastable = gradient.reshape(std::move(expanded_shape));
            }
            auto expanded = Tensor::ones(parents[0].shape()) * broadcastable;
            return detail::autograd::GradList{
                std::optional<Tensor>{std::move(expanded)}};
        }

        detail::autograd::BackwardFn make_sum_backward(
            const std::optional<Index> dim,
            const bool keepdim)
        {
            return [dim, keepdim](
                       const Tensor &gradient,
                       const detail::autograd::TensorSpan parents,
                       const detail::autograd::TensorSpan saved_tensors)
            { return sum_backward(gradient, parents, saved_tensors, dim, keepdim); };
        }

    } // namespace

    Tensor sum(const Tensor &tensor, const std::optional<Index> dim, const bool keepdim)
    {
        const auto output_shape = detail::shape::reduce(tensor.shape(), dim, keepdim);
        detail::autograd::OperationContext context{tensor};
        auto result = Tensor::zeros(output_shape, context.requires_grad());
        detail::kernel::reduce_sum(
            detail::kernel::TensorViewAccess::view(tensor),
            detail::kernel::TensorViewAccess::mutable_view(result),
            dim,
            keepdim);

        context.record(result, "sum", make_sum_backward(dim, keepdim));
        return result;
    }

} // namespace minitensor
