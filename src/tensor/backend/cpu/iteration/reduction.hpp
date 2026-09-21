#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    // Describes the CPU traversal topology of a reduction independently of
    // its value type and reduction operation.
    class ReductionPlan final
    {
    private:
        struct Dimension final
        {
            Shape::size_type extent;
            Layout::stride_type step;
            Layout::offset_type reset;
        };

    public:
        using offset_type = Layout::offset_type;
        using stride_type = Layout::stride_type;
        using size_type = Shape::size_type;

        struct Run final
        {
            offset_type offset;
            stride_type stride;
            size_type size;
        };

        // A callback-scoped view of the input values reduced into one output.
        // Runs are maximal single-stride suffixes of the reduced dimensions.
        class ReductionRange final
        {
        public:
            [[nodiscard]] size_type size() const noexcept
            {
                return size_;
            }

            template <typename Function>
            void for_each_run(Function &&function) const
            {
                if (size_ == 0)
                {
                    return;
                }

                std::fill(indices_.begin(), indices_.end(), size_type{0});
                offset_type offset = initial_offset_;
                const size_type run_count = size_ / run_size_;

                for (size_type run = 0; run < run_count; ++run)
                {
                    std::invoke(function, Run{offset, run_stride_, run_size_});
                    if (run + 1 < run_count)
                    {
                        advance(indices_, dimensions_, offset);
                    }
                }
            }

        private:
            friend class ReductionPlan;

            ReductionRange(
                offset_type initial_offset,
                size_type size,
                size_type run_size,
                stride_type run_stride,
                std::span<const Dimension> dimensions,
                std::span<size_type> indices) noexcept
                : initial_offset_{initial_offset},
                  size_{size},
                  run_size_{run_size},
                  run_stride_{run_stride},
                  dimensions_{dimensions},
                  indices_{indices}
            {
            }

            static void advance(
                std::span<size_type> indices,
                std::span<const Dimension> dimensions,
                offset_type &offset) noexcept
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

            offset_type initial_offset_;
            size_type size_;
            size_type run_size_;
            stride_type run_stride_;
            std::span<const Dimension> dimensions_;
            std::span<size_type> indices_;
        };

        ReductionPlan(
            const Shape &input_shape,
            const Layout &input_layout,
            std::span<const size_type> axes)
            : initial_offset_{input_layout.offset()}
        {
            if (input_layout.rank() != input_shape.rank())
            {
                throw std::invalid_argument{
                    "input layout rank does not match reduction shape rank"};
            }

            std::vector<bool> is_reduced(input_shape.rank(), false);
            for (const size_type axis : axes)
            {
                if (axis >= input_shape.rank())
                {
                    throw std::invalid_argument{"reduction axis is out of range"};
                }
                if (is_reduced[axis])
                {
                    throw std::invalid_argument{"reduction axes contain a duplicate"};
                }
                is_reduced[axis] = true;
            }

            outer_dimensions_.reserve(input_shape.rank() - axes.size());
            reduction_dimensions_.reserve(axes.size());

            for (size_type axis = 0; axis < input_shape.rank(); ++axis)
            {
                const size_type extent = static_cast<size_type>(input_shape[axis]);
                size_type &count = is_reduced[axis] ? reduction_size_ : output_size_;
                count = checked_product(count, extent);

                // Singleton dimensions do not participate in offset traversal.
                // Zero-sized dimensions also cannot be traversed; their zero
                // count already determines the corresponding iteration size.
                if (extent <= 1)
                {
                    continue;
                }

                const stride_type step = input_layout.stride(axis);
                const Dimension dimension{
                    extent,
                    step,
                    step * static_cast<offset_type>(extent - 1)};

                if (is_reduced[axis])
                {
                    reduction_dimensions_.push_back(dimension);
                }
                else
                {
                    outer_dimensions_.push_back(dimension);
                }
            }

            coalesce_reduction_suffix();
        }

        [[nodiscard]] size_type output_size() const noexcept
        {
            return output_size_;
        }

        [[nodiscard]] size_type reduction_size() const noexcept
        {
            return reduction_size_;
        }

        // Calls function(output_linear_index, reduction_range). The range is
        // valid only for the duration of the callback.
        template <typename Function>
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
            offset_type outer_offset = initial_offset_;

            for (size_type output_linear = 0;
                 output_linear < output_size_;
                 ++output_linear)
            {
                const ReductionRange range{
                    outer_offset,
                    reduction_size_,
                    reduction_run_size_,
                    reduction_run_stride_,
                    reduction_dimensions_,
                    reduction_indices};
                std::invoke(function, output_linear, range);

                if (output_linear + 1 < output_size_)
                {
                    ReductionRange::advance(
                        outer_indices, outer_dimensions_, outer_offset);
                }
            }
        }

    private:
        [[nodiscard]] static size_type checked_product(size_type lhs, size_type rhs)
        {
            if (rhs != 0 && lhs > std::numeric_limits<size_type>::max() / rhs)
            {
                throw std::overflow_error{"reduction iteration size overflow"};
            }
            return lhs * rhs;
        }

        void coalesce_reduction_suffix() noexcept
        {
            if (reduction_size_ == 0 || reduction_dimensions_.empty())
            {
                return;
            }

            const Dimension &innermost = reduction_dimensions_.back();
            reduction_run_size_ = innermost.extent;
            reduction_run_stride_ = innermost.step;
            std::size_t run_axis_start = reduction_dimensions_.size() - 1;

            while (run_axis_start > 0)
            {
                if (reduction_run_size_ >
                    static_cast<size_type>(std::numeric_limits<offset_type>::max()))
                {
                    break;
                }

                const offset_type signed_run_size =
                    static_cast<offset_type>(reduction_run_size_);
                const Dimension &outer = reduction_dimensions_[run_axis_start - 1];
                if (outer.step % signed_run_size != 0 ||
                    outer.step / signed_run_size != reduction_run_stride_)
                {
                    break;
                }

                reduction_run_size_ *= outer.extent;
                --run_axis_start;
            }

            reduction_dimensions_.resize(run_axis_start);
        }

        offset_type initial_offset_{};
        size_type output_size_{1};
        size_type reduction_size_{1};
        size_type reduction_run_size_{1};
        stride_type reduction_run_stride_{};
        std::vector<Dimension> outer_dimensions_;
        std::vector<Dimension> reduction_dimensions_;
    };
}
