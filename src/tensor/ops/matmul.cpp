#include "matmul.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
    namespace detail
    {
        std::string_view MatmulPrimitive::name() const noexcept
        {
            return "matmul";
        }

        TensorSpec MatmulPrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 2)
            {
                throw std::invalid_argument{"matmul expects 2 input tensors"};
            }

            const TensorSpec &lhs = inputs[0];
            const TensorSpec &rhs = inputs[1];

            if (lhs.dtype != rhs.dtype)
            {
                throw std::invalid_argument{"matmul requires matching dtypes"};
            }
            if (lhs.device != rhs.device)
            {
                throw std::invalid_argument{"matmul requires input tensors on the same device"};
            }
        }

        std::vector<std::optional<Tensor>> MatmulPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
        {
            if (inputs.size() != 2)
            {
                throw std::logic_error{"multiply VJP expects 2 inputs"};
            }
        }
    }

    Tensor matmul(const Tensor &lhs, const Tensor &rhs)
    {
        std::array<detail::ValueRef, 2> inputs{detail::TensorAccess::value(lhs), detail::TensorAccess::value(rhs)};
        detail::ValueRef output = detail::apply_operation(std::make_unique<detail::MatmulPrimitive>(), inputs);
        return detail::TensorAccess::make(std::move(output));
    }
}
