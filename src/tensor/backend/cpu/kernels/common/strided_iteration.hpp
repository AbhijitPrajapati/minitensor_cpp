#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    // CPU-only fallback traversal for elementwise and copy kernels. Kernel
    // entry points should select their contiguous and scalar-broadcast paths
    // before constructing this plan.
    template <std::size_t OperandCount>
    class StridedIteration final
    {
        static_assert(OperandCount > 0);

    public:
        using offset_type = Layout::offset_type;
        using stride_type = Layout::stride_type;
        using size_type = Shape::size_type;
        using Offsets = std::array<offset_type, OperandCount>;
        using Strides = std::array<stride_type, OperandCount>;

        StridedIteration(const Shape &shape, const std::array<Layout, OperandCount> &layouts)
            : numel_(shape.numel())
        {
            for (std::size_t operand = 0; operand < OperandCount; ++operand)
            {
                if (layouts[operand].rank() != shape.rank())
                {
                    throw std::invalid_argument{"layout rank does not match iteration shape rank"};
                }
                initial_offsets_[operand] = layouts[operand].offset();
            }

            if (numel_ == 0)
            {
                return;
            }

            axes_.reserve(shape.rank());
            for (size_type axis_index = 0; axis_index < shape.rank(); ++axis_index)
            {
                const Extent extent = shape[axis_index];
                if (extent == 1)
                {
                    continue;
                }

                Axis axis{};
                axis.extent = extent;
                for (std::size_t operand = 0; operand < OperandCount; ++operand)
                {
                    axis.steps[operand] = layouts[operand].stride(axis_index);
                    axis.resets[operand] =
                        axis.steps[operand] * static_cast<offset_type>(extent - 1);
                }
                axes_.push_back(axis);
            }
        }

        [[nodiscard]] size_type numel() const noexcept
        {
            return numel_;
        }

        // Calls function(linear_index, offsets, strides, run_size). The
        // templated callback is invoked once per coalesced run and is inlined;
        // it is not a type-erased callback in the per-element hot loop.
        template <typename Function>
        void for_each_run(Function &&function) const
        {
            if (numel_ == 0)
            {
                return;
            }

            Offsets offsets = initial_offsets_;
            if (axes_.empty())
            {
                std::invoke(function, size_type{0}, offsets, Strides{}, size_type{1});
                return;
            }

            const Axis &run_axis = axes_.back();
            size_type run_size = static_cast<size_type>(run_axis.extent);
            std::size_t run_axis_start = axes_.size() - 1;

            // Adjacent axes can form one run only if every operand advances as
            // a single strided sequence across the boundary.
            while (run_axis_start > 0)
            {
                if (run_size > static_cast<size_type>(std::numeric_limits<offset_type>::max()))
                {
                    break;
                }

                const offset_type signed_run_size = static_cast<offset_type>(run_size);
                const Axis &outer_axis = axes_[run_axis_start - 1];
                bool can_coalesce = true;
                for (std::size_t operand = 0; operand < OperandCount; ++operand)
                {
                    const offset_type outer_step = outer_axis.steps[operand];
                    if (outer_step % signed_run_size != 0 ||
                        outer_step / signed_run_size != run_axis.steps[operand])
                    {
                        can_coalesce = false;
                        break;
                    }
                }

                if (!can_coalesce)
                {
                    break;
                }

                run_size *= static_cast<size_type>(outer_axis.extent);
                --run_axis_start;
            }

            std::vector<Extent> outer_indices(run_axis_start, Extent{0});
            for (size_type linear = 0; linear < numel_; linear += run_size)
            {
                std::invoke(function, linear, offsets, run_axis.steps, run_size);

                if (linear + run_size == numel_)
                {
                    break;
                }

                for (std::size_t i = run_axis_start; i > 0; --i)
                {
                    const std::size_t axis_index = i - 1;
                    const Axis &axis = axes_[axis_index];
                    Extent &index = outer_indices[axis_index];

                    if (index + 1 < axis.extent)
                    {
                        ++index;
                        for (std::size_t operand = 0; operand < OperandCount; ++operand)
                        {
                            offsets[operand] += axis.steps[operand];
                        }
                        break;
                    }

                    index = 0;
                    for (std::size_t operand = 0; operand < OperandCount; ++operand)
                    {
                        offsets[operand] -= axis.resets[operand];
                    }
                }
            }
        }

    private:
        struct Axis final
        {
            Extent extent{};
            Strides steps{};
            Offsets resets{};
        };

        size_type numel_{};
        Offsets initial_offsets_{};
        std::vector<Axis> axes_;
    };

    [[nodiscard]] inline bool is_single_value_layout(
        const Shape &shape,
        const Layout &layout) noexcept
    {
        assert(shape.rank() == layout.rank());
        for (Shape::size_type axis = 0; axis < shape.rank(); ++axis)
        {
            if (shape[axis] > 1 && layout.stride(axis) != 0)
            {
                return false;
            }
        }
        return true;
    }
}
