#include "matmul.hpp"

#include <algorithm>
#include <stdexcept>

#include <minitensor/types.hpp>

#include "tensor/core/shape_inference.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
	namespace
	{
		// Retrives the stride for an input batch axis
		MatmulPlan::stride_type aligned_batch_stride(
			const Shape& shape,
			const Layout& layout,
			MatmulPlan::size_type batch_rank,
			MatmulPlan::size_type output_batch_rank,
			MatmulPlan::size_type output_axis) noexcept
		{
			// If axis was appended as a leading dimension during right alignment, then the stride is 0
			const MatmulPlan::size_type leading_axes = output_batch_rank - batch_rank;
			if (output_axis < leading_axes)
			{
				return 0;
			}

			// otherwise, return the corresponding stride
			const MatmulPlan::size_type input_axis = output_axis - leading_axes;
			return shape[input_axis] == 1 ? 0 : layout.stride(input_axis);
		}
	}

	MatmulPlan::MatmulPlan(
		const Shape& lhs_shape,
		const Layout& lhs_layout,
		const Shape& rhs_shape,
		const Layout& rhs_layout,
		const Shape& output_shape,
		const Layout& output_layout)
	{
		if (lhs_layout.rank() != lhs_shape.rank() ||
			rhs_layout.rank() != rhs_shape.rank() ||
			output_layout.rank() != output_shape.rank())
		{
			throw std::invalid_argument{ "matmul shape and layout ranks do not match" };
		}

		if (matmul_output_shape(lhs_shape, rhs_shape) != output_shape)
		{
			throw std::invalid_argument{ "matmul output shape does not match input shapes" };
		}

		const bool lhs_is_vector = lhs_shape.rank() == 1;
		const bool rhs_is_vector = rhs_shape.rank() == 1;
		const size_type lhs_batch_rank = lhs_is_vector ? 0 : lhs_shape.rank() - 2;
		const size_type rhs_batch_rank = rhs_is_vector ? 0 : rhs_shape.rank() - 2;
		const size_type output_batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);

		row_count_ = static_cast<size_type>(
			lhs_is_vector ? Extent{ 1 } : lhs_shape[lhs_shape.rank() - 2]);
		contraction_size_ = static_cast<size_type>(lhs_shape[lhs_shape.rank() - 1]);
		column_count_ = static_cast<size_type>(
			rhs_is_vector ? Extent{ 1 } : rhs_shape[rhs_shape.rank() - 1]);

		lhs_strides_ = MatrixStrides{
			lhs_is_vector ? stride_type{0} : lhs_layout.stride(lhs_shape.rank() - 2),
			lhs_layout.stride(lhs_shape.rank() - 1) };
		rhs_strides_ = MatrixStrides{
			rhs_is_vector ? rhs_layout.stride(0) : rhs_layout.stride(rhs_shape.rank() - 2),
			rhs_is_vector ? stride_type{0} : rhs_layout.stride(rhs_shape.rank() - 1) };
		output_strides_ = MatrixStrides{
			lhs_is_vector ? stride_type{0} : output_layout.stride(output_batch_rank),
			rhs_is_vector
				? stride_type{0}
				: output_layout.stride(output_batch_rank + (lhs_is_vector ? 0 : 1)) };

		initial_offsets_ = BatchOffsets{
			lhs_layout.offset(),
			rhs_layout.offset(),
			output_layout.offset() };

		dimensions_.reserve(output_batch_rank);
		for (size_type output_axis = 0; output_axis < output_batch_rank; ++output_axis)
		{
			const size_type extent = static_cast<size_type>(output_shape[output_axis]);
			batch_count_ *= extent;

			if (extent <= 1)
			{
				continue;
			}

			const BatchOffsets steps{
				aligned_batch_stride(lhs_shape, lhs_layout, lhs_batch_rank, output_batch_rank, output_axis),
				aligned_batch_stride(rhs_shape, rhs_layout, rhs_batch_rank, output_batch_rank, output_axis),
				output_layout.stride(output_axis) };
			const auto reset_scale = static_cast<offset_type>(extent - 1);

			const BatchOffsets resets{
				steps.lhs * reset_scale,
				steps.rhs * reset_scale,
				steps.output * reset_scale };

			dimensions_.push_back(Dimension{
				extent,
				steps,
				resets });
		}
	}

	MatmulPlan::size_type MatmulPlan::batch_count() const noexcept
	{
		return batch_count_;
	}

	MatmulPlan::size_type MatmulPlan::row_count() const noexcept
	{
		return row_count_;
	}

	MatmulPlan::size_type MatmulPlan::contraction_size() const noexcept
	{
		return contraction_size_;
	}

	MatmulPlan::size_type MatmulPlan::column_count() const noexcept
	{
		return column_count_;
	}

	const MatmulPlan::MatrixStrides& MatmulPlan::lhs_strides() const noexcept
	{
		return lhs_strides_;
	}

	const MatmulPlan::MatrixStrides& MatmulPlan::rhs_strides() const noexcept
	{
		return rhs_strides_;
	}

	const MatmulPlan::MatrixStrides& MatmulPlan::output_strides() const noexcept
	{
		return output_strides_;
	}
}
