#include "multiply.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops.hpp>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/core/broadcast_shape.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    std::string_view MultiplyPrimitive::name() const noexcept
    {
        return "multiply";
    }

    TensorSpec MultiplyPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        if (inputs.size() != 2)
        {
            throw std::invalid_argument{"multiply expects 2 input tensors"};
        }

        const TensorSpec &lhs = inputs[0];
        const TensorSpec &rhs = inputs[1];

        if (lhs.dtype != rhs.dtype)
        {
            throw std::invalid_argument{"multiply requires matching dtypes"};
        }
        if (lhs.device != rhs.device)
        {
            throw std::invalid_argument{"multiply requires input tensors on the same device"};
        }

        Shape output_shape = broadcast_shape(lhs.shape, rhs.shape);
        return TensorSpec{std::move(output_shape), lhs.dtype, lhs.device};
    }

    std::vector<std::optional<Tensor>> MultiplyPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 2)
        {
            throw std::logic_error{"multiply VJP expects 2 inputs"};
        }

        Tensor lhs_cotangent = output_cotangent * inputs[1];
        Tensor rhs_cotangent = output_cotangent * inputs[0];
        return {
            reduce_to_shape(lhs_cotangent, inputs[0].shape()),
            reduce_to_shape(rhs_cotangent, inputs[1].shape())};
    }
}
