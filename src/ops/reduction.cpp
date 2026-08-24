#include "minitensor/ops.hpp"

#include "autograd/node.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <utility>

namespace minitensor
{

    Tensor sum(const Tensor &tensor, const std::optional<Index> dim, const bool keepdim)
    {
        const auto output_shape = tensor.shape().reduced(dim, keepdim);
        auto result = detail::make_contiguous_tensor(output_shape, tensor.requires_grad());
        detail::reduce_sum(detail::read_arg(tensor), detail::write_arg(result), dim, keepdim);

        if (tensor.requires_grad())
        {
            const auto input_shape = tensor.shape();
            detail::set_history(
                result,
                "sum",
                {tensor},
                [input_shape, dim, keepdim](const Tensor &gradient)
                {
                    auto broadcastable = gradient.detach();
                    if (dim.has_value() && !keepdim)
                    {
                        auto expanded_shape = gradient.shape().with_inserted_axis(*dim);
                        broadcastable = gradient.reshape(std::move(expanded_shape));
                    }
                    auto expanded = Tensor::ones(input_shape) * broadcastable;
                    return detail::GradList{std::optional<Tensor>{std::move(expanded)}};
                });
        }
        return result;
    }

} // namespace minitensor
