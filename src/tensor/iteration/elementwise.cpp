#include "elementwise.hpp"

#include <span>
#include <stdexcept>
#include <cstddef>
#include <utility>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/core/checked_arithmetic.hpp"

namespace minitensor::detail
{
	ElementwisePlan::ElementwisePlan(const Shape &shape, std::span<const Layout> layouts) : numel_(shape.numel())
	{
		if (layouts.size() == 0)
		{
			throw std::invalid_argument{"must provide atleast one layout"};
		}

		initial_offsets_.reserve(layouts.size());
		for (const Layout &layout : layouts)
		{
			if (layout.rank() != shape.rank())
			{
				throw std::invalid_argument{"layout rank does not match shape rank"};
			}
			initial_offsets_.push_back(layout.offset());
		}

		if (numel_ == 0)
		{
			return;
		}

		axes_.reserve(shape.rank());

		for (Shape::size_type axis_idx = 0; axis_idx < shape.rank(); ++axis_idx)
		{
			const Extent extent = shape[axis_idx];
			if (extent == 1)
			{
				continue;
			}

			AxisPlan axis;
			axis.extent = extent;
			axis.steps.reserve(layouts.size());
			axis.resets.reserve(layouts.size());

			for (const Layout &layout : layouts)
			{
				const offset_type step = layout.stride(axis_idx);
				axis.steps.push_back(step);
				const auto checked_reset = checked_multiply(step, static_cast<offset_type>(extent - 1));
				axis.resets.push_back(checked_reset);
			}

			axes_.push_back(std::move(axis));
		}
	}

	ElementwisePlan::size_type ElementwisePlan::numel() const noexcept
	{
		return numel_;
	}

	std::size_t ElementwisePlan::layout_count() const noexcept
	{
		return initial_offsets_.size();
	}
}
