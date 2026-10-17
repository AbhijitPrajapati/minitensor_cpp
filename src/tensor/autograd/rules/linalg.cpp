#include "registrations.hpp"

#include <stdexcept>

#include <minitensor/ops/linalg.hpp>
#include <minitensor/ops/manipulation.hpp>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/autograd/vjp_registry.hpp"
#include "tensor/primitives/linalg/matmul.hpp"

namespace minitensor::detail
{
	void register_linalg_vjps(VjpRegistry& registry)
	{
		registry.register_rule<MatmulPrimitive>(
			[](const MatmulPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 2)
				{
					throw std::logic_error{ "matmul VJP expects 2 inputs" };
				}

				const Tensor& lhs = context.inputs[0];
				const Tensor& rhs = context.inputs[1];
				const bool lhs_was_vector = lhs.rank() == 1;
				const bool rhs_was_vector = rhs.rank() == 1;
				Tensor lhs_matrix = lhs_was_vector ? unsqueeze(lhs, -2) : lhs;
				Tensor rhs_matrix = rhs_was_vector ? unsqueeze(rhs, -1) : rhs;

				Tensor cotangent_matrix = context.output_cotangent;
				if (rhs_was_vector)
				{
					cotangent_matrix = unsqueeze(cotangent_matrix, -1);
				}
				if (lhs_was_vector)
				{
					cotangent_matrix = unsqueeze(cotangent_matrix, -2);
				}

				Tensor lhs_cotangent = reduce_to_shape(
					matmul(cotangent_matrix, transpose(rhs_matrix, -1, -2)),
					lhs_matrix.shape());
				Tensor rhs_cotangent = reduce_to_shape(
					matmul(transpose(lhs_matrix, -1, -2), cotangent_matrix),
					rhs_matrix.shape());

				if (lhs_was_vector)
				{
					lhs_cotangent = squeeze(lhs_cotangent, -2);
				}
				if (rhs_was_vector)
				{
					rhs_cotangent = squeeze(rhs_cotangent, -1);
				}

				return VjpContributions{
					std::move(lhs_cotangent), std::move(rhs_cotangent) };
			});
	}
}
