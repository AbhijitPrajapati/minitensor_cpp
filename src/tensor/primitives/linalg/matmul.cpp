#include "matmul.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops/linalg.hpp>
#include <minitensor/ops/manipulation.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/autograd/reduce_to_shape.hpp"

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

        const Shape::size_type lhs_rank = lhs.shape.rank();
        const Shape::size_type rhs_rank = rhs.shape.rank();
        if (lhs_rank == 0 || lhs_rank > 2 || rhs_rank == 0 || rhs_rank > 2)
        {
            throw std::invalid_argument{
                "matmul currently requires rank-1 or rank-2 input tensors"};
        }

        const Extent lhs_contraction_extent = lhs.shape[lhs_rank - 1];
        const Extent rhs_contraction_extent = rhs.shape[rhs_rank == 1 ? 0 : rhs_rank - 2];
        if (lhs_contraction_extent != rhs_contraction_extent)
        {
            throw std::invalid_argument{"matmul contraction dimensions must match"};
        }

        std::vector<Extent> output_dimensions;
        output_dimensions.reserve(lhs_rank + rhs_rank - 2);
        if (lhs_rank == 2)
        {
            output_dimensions.push_back(lhs.shape[0]);
        }
        if (rhs_rank == 2)
        {
            output_dimensions.push_back(rhs.shape[1]);
        }

        return TensorSpec{
            Shape{std::move(output_dimensions)},
            lhs.dtype,
            lhs.device};
    }

    std::vector<std::optional<Tensor>> MatmulPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 2)
        {
            throw std::logic_error{"matmul VJP expects 2 inputs"};
        }

        Tensor lhs_cotangent = matmul(output_cotangent, transpose(inputs[1], -1, -2));
        Tensor rhs_cotangent = matmul(transpose(inputs[0], -1, -2), output_cotangent);
        return {
            reduce_to_shape(lhs_cotangent, inputs[0].shape()),
            reduce_to_shape(rhs_cotangent, inputs[1].shape())};
    }
}
