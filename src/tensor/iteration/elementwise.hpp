#pragma once

#include <cstddef>
#include <concepts>
#include <functional>
#include <limits>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	template <typename Function>
	concept ElementwiseRunFunction = std::invocable<
		Function &,
		Shape::size_type,
		std::span<const Layout::offset_type>,
		std::span<const Layout::stride_type>,
		Shape::size_type>;

	class ElementwisePlan final
	{
	public:
		using offset_type = Layout::offset_type;
		using stride_type = Layout::stride_type;
		using size_type = Shape::size_type;

		ElementwisePlan(const Shape &shape, std::span<const Layout> layouts);

		[[nodiscard]] size_type numel() const noexcept;
		[[nodiscard]] std::size_t layout_count() const noexcept;

		// Calls function(linear_index, layout_offsets, layout_strides, run_size)
		// for each run along the innermost non-singleton dimension. Offsets point
		// to the first logical element in the run and strides advance each layout.
		template <ElementwiseRunFunction Function>
		void for_each_run(Function &&function) const;

	private:
		struct AxisPlan
		{
			Extent extent;
			// amount added to each layout when the axis advances by one
			std::vector<offset_type> steps;
			// amount subtracted when the axis wraps to 0
			std::vector<offset_type> resets;
		};

		Shape::size_type numel_;
		std::vector<offset_type> initial_offsets_;
		std::vector<AxisPlan> axes_;
	};

	template <ElementwiseRunFunction Function>
	void ElementwisePlan::for_each_run(Function &&function) const
	{
		if (numel_ == 0)
		{
			return;
		}

		std::vector<offset_type> offsets = initial_offsets_;

		// Scalars and shapes containing only singleton dimensions have one
		// logical element and no meaningful physical stride.
		if (axes_.empty())
		{
			const std::vector<stride_type> strides(layout_count(), stride_type{0});
			std::invoke(function, size_type{0}, offsets, strides, size_type{1});
			return;
		}

		const AxisPlan &run_axis = axes_.back();
		size_type run_size = static_cast<size_type>(run_axis.extent);
		std::size_t run_axis_start = axes_.size() - 1;

		// Fold adjacent dimensions into the run when every layout advances as
		// one physically strided sequence across the dimension boundary.
		while (run_axis_start > 0)
		{
			if (run_size > static_cast<size_type>(std::numeric_limits<offset_type>::max()))
			{
				break;
			}

			const offset_type run_extent = static_cast<offset_type>(run_size);
			const AxisPlan &outer_axis = axes_[run_axis_start - 1];
			bool can_fold = true;
			for (std::size_t layout_idx = 0; layout_idx < layout_count(); ++layout_idx)
			{
				const offset_type outer_step = outer_axis.steps[layout_idx];
				if (outer_step % run_extent != 0 || outer_step / run_extent != run_axis.steps[layout_idx])
				{
					can_fold = false;
					break;
				}
			}

			if (!can_fold)
			{
				break;
			}

			run_size *= static_cast<size_type>(outer_axis.extent);
			--run_axis_start;
		}

		const std::size_t outer_axis_count = run_axis_start;
		std::vector<Extent> outer_indices(outer_axis_count, Extent{0});

		for (size_type linear = 0; linear < numel_; linear += run_size)
		{
			std::invoke(function, linear, offsets, run_axis.steps, run_size);

			if (linear + run_size == numel_)
			{
				break;
			}

			for (std::size_t i = outer_axis_count; i > 0; --i)
			{
				const std::size_t axis_idx = i - 1;
				const AxisPlan &axis = axes_[axis_idx];
				Extent &index = outer_indices[axis_idx];

				if (index + 1 < axis.extent)
				{
					++index;
					for (std::size_t layout_idx = 0; layout_idx < offsets.size(); ++layout_idx)
					{
						offsets[layout_idx] += axis.steps[layout_idx];
					}
					break;
				}

				index = 0;
				for (std::size_t layout_idx = 0; layout_idx < offsets.size(); ++layout_idx)
				{
					offsets[layout_idx] -= axis.resets[layout_idx];
				}
			}
		}
	}

}
