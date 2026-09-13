#pragma once

#include <vector>
#include <span>
#include <concepts>
#include <functional>
#include <algorithm>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
    template <typename Function>
    concept ReductionFunction = std::invocable<Function &, Shape::size_type, Layout::offset_type>;

    class ReductionPlan final
    {
    public:
        using size_type = Shape::size_type;
        using offset_type = Layout::offset_type;
        using stride_type = Layout::stride_type;

        ReductionPlan(const Shape &input_shape, const Layout &input_layout, const Shape &output_shape, std::span<const Shape::size_type> reduced_axes);
        [[nodiscard]] size_type input_numel() const noexcept;
        [[nodiscard]] size_type output_numel() const noexcept;

        // calls function(output_linear_index, physical_input_offset) for each logical element
        template <ReductionFunction Function>
        void for_each(Function &&function) const;

    private:
        struct Dimension
        {
            size_type extent;
            stride_type step;
            offset_type reset;
        };

        static void advance(std::vector<size_type> &indicies, const std::vector<Dimension> &dimensions, offset_type &offset) noexcept;

        std::vector<Dimension> outer_dimensions_;     // preserved axes
        std::vector<Dimension> reduction_dimensions_; // reduction axes

        offset_type input_base_offset_;

        size_type input_numel_;
        size_type output_numel_;

        // number of input values asscociated with a single output value
        size_type reduction_iterations_;
    };

    template <ReductionFunction Function>
    void ReductionPlan::for_each(Function &&function) const
    {
        if (output_numel_ == 0 || reduction_iterations_ == 0)
        {
            return;
        }

        // outer iteration
        std::vector<size_type> outer_indices(outer_dimensions_.size(), size_type{0});

        // reduction iteration
        std::vector<size_type> reduction_indices(reduction_dimensions_.size(), size_type{0});

        offset_type outer_offset = input_base_offset_;

        for (size_type output_linear = 0; output_linear < output_numel_; ++output_linear)
        {
            // start iterating from 0 for reduction
            std::fill(reduction_indices.begin(), reduction_indices.end(), size_type{0});
            offset_type input_offset = outer_offset;
            for (size_type reduction_linear = 0; reduction_linear < reduction_iterations_; ++reduction_linear)
            {
                // multiple input offsets are called with the single output linear
                std::invoke(function, output_linear, input_offset);

                // advance the inner/reduction index
                if (reduction_linear + 1 < reduction_iterations_)
                {
                    advance(reduction_indices, reduction_dimensions_, input_offset);
                }
            }

            // advance the outer index
            if (output_linear + 1 < output_numel_)
            {
                advance(outer_indices, outer_dimensions_, outer_offset);
            }
        }
    }
}