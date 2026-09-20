#include "divide.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops.hpp>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/core/broadcast_shape.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
    namespace detail
    {
        std::string_view DividePrimitive::name() const noexcept
        {
            return "divide";
        }

        TensorSpec DividePrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 2)
            {
                throw std::invalid_argument{"divide expects 2 input tensors"};
            }

            const TensorSpec &lhs = inputs[0];
            const TensorSpec &rhs = inputs[1];

            if (lhs.dtype != rhs.dtype)
            {
                throw std::invalid_argument{"divide requires matching dtypes"};
            }
            if (lhs.device != rhs.device)
            {
                throw std::invalid_argument{"divide requires input tensors on the same device"};
            }

            Shape output_shape = broadcast_shape(lhs.shape, rhs.shape);
            return TensorSpec{std::move(output_shape), lhs.dtype, lhs.device};
        }

        std::vector<std::optional<Tensor>> DividePrimitive::vjp(std::span<const Tensor> inputs, const Tensor &output, const Tensor &output_cotangent) const
        {
            if (inputs.size() != 2)
            {
                throw std::logic_error{"divide VJP expects 2 inputs"};
            }

            Tensor lhs_cotangent = output_cotangent / inputs[1];
            Tensor rhs_cotangent = -(output_cotangent * output / inputs[1]);
            return {
                reduce_to_shape(lhs_cotangent, inputs[0].shape()),
                reduce_to_shape(rhs_cotangent, inputs[1].shape())};
        }
    }

    Tensor operator/(const Tensor &lhs, const Tensor &rhs)
    {
        std::array<detail::ValueRef, 2> inputs{detail::TensorAccess::value(lhs), detail::TensorAccess::value(rhs)};
        detail::ValueRef output = detail::apply_operation(std::make_unique<detail::DividePrimitive>(), inputs);
        return detail::TensorAccess::make(std::move(output));
    }
}
