#include "matmul.hpp"

#include <stdexcept>
#include <vector>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
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

    std::vector<std::optional<Tensor>> MatmulPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &) const
    {
        if (inputs.size() != 2)
        {
            throw std::logic_error{"multiply VJP expects 2 inputs"};
        }
    }
}
