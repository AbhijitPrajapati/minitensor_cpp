#include "add.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/core/shape_inference.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
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

        Shape output_shape = broadcast_shape(lhs.shape, rhs.shape);
        return TensorSpec{std::move(output_shape), lhs.dtype, lhs.device};
    }

    std::vector<std::optional<Tensor>> AddPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 2)
        {
            throw std::logic_error{"add VJP expects 2 inputs"};
        }
        return {
            reduce_to_shape(output_cotangent, inputs[0].shape()),
            reduce_to_shape(output_cotangent, inputs[1].shape())};
    }
}
