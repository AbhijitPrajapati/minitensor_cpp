#include "matmul.hpp"

#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/broadcasting.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	namespace
	{
		Shape matmul_output_shape(const Shape& lhs, const Shape& rhs)
		{
			if (lhs.rank() == 0 || rhs.rank() == 0)
			{
				throw std::invalid_argument{
					"matmul requires inputs with rank 1 or greater" };
			}

			const bool lhs_is_vector = lhs.rank() == 1;
			const bool rhs_is_vector = rhs.rank() == 1;

			const Extent lhs_contraction_extent = lhs[lhs.rank() - 1];
			const Extent rhs_contraction_extent =
				rhs[rhs_is_vector ? 0 : rhs.rank() - 2];
			if (lhs_contraction_extent != rhs_contraction_extent)
			{
				throw std::invalid_argument{
					"matmul contraction dimensions must match" };
			}

			const Shape::size_type lhs_batch_rank =
				lhs_is_vector ? 0 : lhs.rank() - 2;
			const Shape::size_type rhs_batch_rank =
				rhs_is_vector ? 0 : rhs.rank() - 2;

			const auto lhs_batch_extents =
				lhs.dimensions().first(lhs_batch_rank);
			const auto rhs_batch_extents =
				rhs.dimensions().first(rhs_batch_rank);
			std::vector<Extent> output_extents =
				broadcast_extents(lhs_batch_extents, rhs_batch_extents);

			if (!lhs_is_vector)
			{
				output_extents.push_back(lhs[lhs.rank() - 2]);
			}
			if (!rhs_is_vector)
			{
				output_extents.push_back(rhs[rhs.rank() - 1]);
			}

			return Shape{ std::move(output_extents) };
		}
	}

	std::string_view MatmulPrimitive::name() const noexcept
	{
		return "matmul";
	}

	TensorSpec MatmulPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 2)
		{
			throw std::invalid_argument{ "matmul expects 2 input tensors" };
		}

		const TensorSpec& lhs = inputs[0];
		const TensorSpec& rhs = inputs[1];

		if (lhs.dtype != rhs.dtype)
		{
			throw std::invalid_argument{ "matmul requires matching dtypes" };
		}
		if (!is_floating_point(lhs.dtype))
		{
			throw std::invalid_argument{ "matmul requires floating-point inputs" };
		}
		if (lhs.device != rhs.device)
		{
			throw std::invalid_argument{
				"matmul requires input tensors on the same device" };
		}

		Shape output_shape = matmul_output_shape(lhs.shape, rhs.shape);
		return TensorSpec{
			std::move(output_shape),
			lhs.dtype,
			lhs.device };
	}
}
