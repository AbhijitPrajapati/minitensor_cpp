#include "minitensor/ops.hpp"

#include "autograd/node.hpp"
#include "core/shape.hpp"
#include "kernels/kernels.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <utility>

namespace minitensor
{

    Tensor sum(const Tensor &tensor, const std::optional<Index> dim, const bool keepdim)
    {
        const auto output_shape = detail::shape::reduce(tensor.shape(), dim, keepdim);
        auto result = detail::make_contiguous_tensor(output_shape, tensor.requires_grad());
        detail::kernel::reduce_sum(
            detail::kernel::TensorViewAccess::view(tensor),
            detail::kernel::TensorViewAccess::mutable_view(result),
            dim,
            keepdim);

        if (tensor.requires_grad())
        {
            const auto input_shape = tensor.shape();
            detail::autograd::set_history(
                result,
                "sum",
                {tensor},
                [input_shape, dim, keepdim](const Tensor &gradient)
                {
                    auto broadcastable = gradient.detach();
                    if (dim.has_value() && !keepdim)
                    {
                        auto expanded_shape = detail::shape::insert_axis(gradient.shape(), *dim);
                        broadcastable = gradient.reshape(std::move(expanded_shape));
                    }
                    auto expanded = Tensor::ones(input_shape) * broadcastable;
                    return detail::autograd::GradList{
                        std::optional<Tensor>{std::move(expanded)}};
                });
        }
        return result;
    }

} // namespace minitensor
