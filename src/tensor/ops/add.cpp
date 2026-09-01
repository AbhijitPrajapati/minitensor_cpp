#include "add.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>

#include <minitensor/ops.hpp>

#include "tensor/core/broadcast_shape.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
    namespace detail
    {
        std::string_view AddPrimitive::name() const noexcept
        {
            return "add";
        }

        TensorSpec AddPrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 2)
            {
                throw std::invalid_argument{"add expects 2 input tensors"};
            }

            const TensorSpec &lhs = inputs[0];
            const TensorSpec &rhs = inputs[1];

            if (lhs.dtype != rhs.dtype)
            {
                throw std::invalid_argument{"add requires matching dtypes"};
            }
            if (lhs.device != rhs.device)
            {
                throw std::invalid_argument{"add requires input tensors on the same device"};
            }

            Shape output_shape = broadcast_shape(inputs[0].shape, inputs[1].shape);
            return TensorSpec{std::move(output_shape), lhs.dtype, lhs.device};
        }
    }

    Tensor operator+(const Tensor &lhs, const Tensor &rhs)
    {
        std::array<detail::ValueRef, 2> inputs{detail::TensorAccess::value(lhs), detail::TensorAccess::value(rhs)};
        detail::ValueRef output = detail::apply_operation(std::make_unique<detail::AddPrimitive>(), inputs);
        return detail::TensorAccess::make(std::move(output));
    }
}
