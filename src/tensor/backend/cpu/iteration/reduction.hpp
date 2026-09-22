#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    class ReductionPlan final
    {
    private:
        struct Dimension
        {
            Shape::size_type extent;
            Layout::stride_type step;
            Layout::offset_type reset;
        };

    public:
        using size_type = Shape::size_type;
        using offset_type = Layout::offset_type;
        using stride_type = Layout::stride_type;

        struct Run
        {
            offset_type offset;
            stride_type stride;
            size_type size;
        };

        // Each run is a maximal constant-stride suffix of the reduced dimensions.
        class ReductionRange final
        {
        public:
            [[nodiscard]] size_type size() const noexcept;

            template <typename Function>
                requires std::invocable<Function &, const Run &>
            void for_each_run(Function &&function) const
            {
                if (run_count_ == 0)
                {
                    return;
                }

                std::fill(indices_.begin(), indices_.end(), size_type{0});
                offset_type offset = initial_offset_;

                for (size_type run = 0; run < run_count_; ++run)
                {
                    std::invoke(function, Run{offset, run_stride_, run_size_});
                    if (run + 1 < run_count_)
                    {
                        ReductionPlan::advance(indices_, dimensions_, offset);
                    }
                }
            }

        private:
            friend class ReductionPlan;

            ReductionRange(
                offset_type initial_offset,
                size_type size,
                size_type run_size,
                size_type run_count,
                stride_type run_stride,
                std::span<const Dimension> dimensions,
                std::span<size_type> indices) noexcept
                : initial_offset_{initial_offset},
                  size_{size},
                  run_size_{run_size},
                  run_count_{run_count},
                  run_stride_{run_stride},
                  dimensions_{dimensions},
                  indices_{indices}
            {
            }

            offset_type initial_offset_;
            size_type size_;
            size_type run_size_;
            size_type run_count_;
            stride_type run_stride_;
            std::span<const Dimension> dimensions_;
            std::span<size_type> indices_;
        };

        ReductionPlan(
            const Shape &input_shape,
            const Layout &input_layout,
            std::span<const size_type> reduced_axes);

        [[nodiscard]] size_type input_numel() const noexcept;
        [[nodiscard]] size_type output_size() const noexcept;
        [[nodiscard]] size_type reduction_size() const noexcept;

        // Calls function(output_linear_index, reduction_range) once per output.
        template <typename Function>
            requires std::invocable<Function &, size_type, const ReductionRange &>
        void for_each_output(Function &&function) const
        {
            if (output_size_ == 0)
            {
                return;
            }

            std::vector<size_type> outer_indices(
                outer_dimensions_.size(), size_type{0});
            std::vector<size_type> reduction_indices(
                reduction_dimensions_.size(), size_type{0});
            offset_type outer_offset = input_base_offset_;

            for (size_type output_linear = 0;
                 output_linear < output_size_;
                 ++output_linear)
            {
                const ReductionRange range{
                    outer_offset,
                    reduction_size_,
                    reduction_run_size_,
                    reduction_run_count_,
                    reduction_run_stride_,
                    reduction_dimensions_,
                    reduction_indices};
                std::invoke(function, output_linear, range);

                if (output_linear + 1 < output_size_)
                {
                    advance(outer_indices, outer_dimensions_, outer_offset);
                }
            }
        }

    private:
        static void advance(
            std::span<size_type> indices,
            std::span<const Dimension> dimensions,
            offset_type& offset) noexcept
        {
            for (std::size_t i = dimensions.size(); i > 0; --i)
            {
                const std::size_t dimension_index = i - 1;
                const Dimension &dimension = dimensions[dimension_index];
                size_type &index = indices[dimension_index];

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

        [[nodiscard]] static size_type checked_product(size_type lhs, size_type rhs);
        void coalesce_reduction_suffix() noexcept;

        std::vector<Dimension> outer_dimensions_;
        std::vector<Dimension> reduction_dimensions_;

        offset_type input_base_offset_{};
        size_type input_numel_{};
        size_type output_size_{1};
        size_type reduction_size_{1};
        size_type reduction_run_size_{1};
        size_type reduction_run_count_{1};
        stride_type reduction_run_stride_{};
    };
}
