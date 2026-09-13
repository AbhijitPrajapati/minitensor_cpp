#include "reduction.hpp"

#include <vector>
#include <span>
#include <cstdint>
#include <cassert>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	ReductionPlan::size_type ReductionPlan::input_numel() const noexcept
	{
		return input_numel_;
	}

	ReductionPlan::size_type ReductionPlan::output_numel() const noexcept
	{
		return output_numel_;
	}

	void ReductionPlan::advance(std::vector<size_type> &indicies, const std::vector<Dimension> &dimensions, offset_type &offset) noexcept
	{

		// advance the multi-dimensional index along with the phyisical offset

		for (size_type axis = dimensions.size(); axis-- > 0;)
		{
			const Dimension &dimension = dimensions[axis];
			size_type &index = indicies[axis];

			++index;

			if (index < dimension.extent)
			{
				offset += dimension.step;
				return;
			}

			index = 0;
			offset -= dimension.reset;
		}
	}

	ReductionPlan::ReductionPlan(const Shape &input_shape, const Layout &input_layout, const Shape &output_shape, std::span<const Shape::size_type> reduced_axes)
		: input_base_offset_(input_layout.offset()), input_numel_(input_shape.numel()), output_numel_(output_shape.numel())
	{
		const size_type input_rank = input_shape.rank();

		// track reduced dimensions
		std::vector<bool> is_reduced(input_rank, false);
		for (const Shape::size_type axis : reduced_axes)
		{
			is_reduced[axis] = true;
		}

		const std::size_t num_reduced = reduced_axes.size();
		reduction_dimensions_.reserve(num_reduced);
		outer_dimensions_.reserve(input_rank - num_reduced);

		// construct and sort dimensions into outer (preserved) and reduced
		for (size_type axis = 0; axis < input_rank; ++axis)
		{
			const size_type extent = static_cast<size_type>(input_shape[axis]);
			const stride_type stride = input_layout.stride(axis);
			const offset_type reset = extent == 0 ? offset_type{0} : stride * static_cast<offset_type>(extent - 1);
			Dimension dimension{extent, stride, reset};
			if (is_reduced[axis])
			{
				reduction_dimensions_.push_back(dimension);
			}
			else
			{
				outer_dimensions_.push_back(dimension);
			}
		}

		// set the number of input elements that correspond to a single output element
		if (output_numel_ == 0)
		{
			reduction_iterations_ = 0;
		}
		else
		{
			assert(input_numel_ % output_numel_ == 0);
			reduction_iterations_ = input_numel_ / output_numel_;
		}
	}
}