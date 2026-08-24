#include "minitensor/ops.hpp"

#include "autograd/node.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <stdexcept>
#include <utility>

namespace minitensor
{

    Tensor Tensor::sum(
        const std::optional<Index> dim,
        const bool keepdim) const
    {
        if (dim.has_value() && (*dim < 0 || *dim >= rank()))
        {
            throw std::out_of_range("sum dimension is out of range");
        }

        Shape output_shape;
        if (!dim.has_value())
        {
            if (keepdim)
            {
                output_shape.assign(static_cast<std::size_t>(rank()), 1);
            }
        }
        else
        {
            output_shape = shape();
            if (keepdim)
            {
                output_shape[static_cast<std::size_t>(*dim)] = 1;
            }
            else
            {
                output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(*dim));
            }
        }

        auto result = detail::make_contiguous_tensor(output_shape, requires_grad());
        detail::reduce_sum(detail::read_arg(*this), detail::write_arg(result), dim, keepdim);

        if (requires_grad())
        {
            const auto input_shape = shape();
            detail::set_history(
                result,
                "sum",
                {*this},
                [input_shape, dim, keepdim](const Tensor &gradient)
                {
                    auto broadcastable = gradient.detach();
                    if (dim.has_value() && !keepdim)
                    {
                        auto expanded_shape = gradient.shape();
                        expanded_shape.insert(
                            expanded_shape.begin() + static_cast<std::ptrdiff_t>(*dim), 1);
                        broadcastable = gradient.reshape(std::move(expanded_shape));
                    }
                    auto expanded = Tensor::ones(input_shape) * broadcastable;
                    return detail::GradList{std::optional<Tensor>{std::move(expanded)}};
                });
        }
        return result;
    }

} // namespace minitensor
