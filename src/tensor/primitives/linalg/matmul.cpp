#include "matmul.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops/linalg.hpp>
#include <minitensor/ops/manipulation.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/core/shape_inference.hpp"

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

        Shape output_shape = matmul_output_shape(lhs.shape, rhs.shape);
        return TensorSpec{
            std::move(output_shape),
            lhs.dtype,
            lhs.device};
    }

    std::vector<std::optional<Tensor>> MatmulPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 2)
        {
            throw std::logic_error{"matmul VJP expects 2 inputs"};
        }

        const Tensor &lhs = inputs[0];
        const Tensor &rhs = inputs[1];

        // promote vectors to matrices
        // lhs and rhs will have atleast two dimensions now
        const bool lhs_was_vector = lhs.rank() == 1;
        const bool rhs_was_vector = rhs.rank() == 1;
        Tensor lhs_matrix = lhs_was_vector ? unsqueeze(lhs, -2) : lhs;
        Tensor rhs_matrix = rhs_was_vector ? unsqueeze(rhs, -1) : rhs;

        // add in synthetic dimensions from vector operands
        Tensor cotangent_mat = output_cotangent;
        if (rhs_was_vector)
        {
            cotangent_mat = unsqueeze(cotangent_mat, -1);
        }
        if (lhs_was_vector)
        {
            cotangent_mat = unsqueeze(cotangent_mat, -2);
        }

        Tensor lhs_cotangent = reduce_to_shape(
            matmul(cotangent_mat, transpose(rhs_matrix, -1, -2)),
            lhs_matrix.shape());
        Tensor rhs_cotangent = reduce_to_shape(
            matmul(transpose(lhs_matrix, -1, -2), cotangent_mat),
            rhs_matrix.shape());

        // remove synthetic dimensions from vector operands
        if (lhs_was_vector)
        {
            lhs_cotangent = squeeze(lhs_cotangent, -2);
        }
        if (rhs_was_vector)
        {
            rhs_cotangent = squeeze(rhs_cotangent, -1);
        }

        return {lhs_cotangent, rhs_cotangent};
    }
}
