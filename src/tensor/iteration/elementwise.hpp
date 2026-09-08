#pragma once

#include <cstddef>
#include <concepts>
#include <functional>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	template <typename Function>
	concept ElementwiseFunction = std::invocable<Function &, Shape::size_type, std::span<const Layout::offset_type>>;

	class ElementwisePlan final
	{
	public:
		using offset_type = Layout::offset_type;
		using size_type = Shape::size_type;

		ElementwisePlan(const Shape &shape, std::span<const Layout> layouts);

		[[nodiscard]] size_type numel() const noexcept;
		[[nodiscard]] std::size_t layout_count() const noexcept;

		// calls function(linear_index, layout_offsets) for each logical element
		template <ElementwiseFunction Function>
		void for_each(Function &&function) const;

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

	template <ElementwiseFunction Function>
	void ElementwisePlan::for_each(Function &&function) const
	{
		if (numel_ == 0)
		{
			return;
		}

		std::vector<offset_type> offsets = initial_offsets_;
		std::vector<Extent> indices(axes_.size(), Extent{0});

		for (size_type linear = 0; linear < numel_; ++linear)
		{
			std::invoke(
				function,
				linear,
				offsets);

			if (linear + 1 == numel_)
			{
				break;
			}

			for (std::size_t i = axes_.size(); i > 0; --i)
			{
				const std::size_t axis_idx = i - 1;
				const AxisPlan &axis = axes_[axis_idx];
				Extent &index = indices[axis_idx];

				// no wrap
				if (index + 1 < axis.extent)
				{
					++index;
					for (std::size_t layout_idx = 0; layout_idx < offsets.size(); ++layout_idx)
					{
						offsets[layout_idx] += axis.steps[layout_idx];
					}
					break;
				}

				// wrap -> subtract the reset values
				index = 0;
				for (std::size_t layout_idx = 0; layout_idx < offsets.size(); ++layout_idx)
				{
					offsets[layout_idx] -= axis.resets[layout_idx];
				}
			}
		}
	}

}
