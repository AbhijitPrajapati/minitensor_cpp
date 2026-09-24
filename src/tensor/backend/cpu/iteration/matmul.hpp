#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    class MatmulPlan final
    {
    public:
        using size_type = Shape::size_type;
        using offset_type = Layout::offset_type;
        using stride_type = Layout::stride_type;

        struct BatchOffsets final
        {
            offset_type lhs;
            offset_type rhs;
            offset_type output;

            friend bool operator==(const BatchOffsets &, const BatchOffsets &) = default;
        };

        struct MatrixStrides final
        {
            stride_type row;
            stride_type column;

            friend bool operator==(const MatrixStrides &, const MatrixStrides &) = default;
        };

        MatmulPlan(
            const Shape &lhs_shape,
            const Layout &lhs_layout,
            const Shape &rhs_shape,
            const Layout &rhs_layout,
            const Shape &output_shape,
            const Layout &output_layout);

        [[nodiscard]] size_type batch_count() const noexcept;
        [[nodiscard]] size_type row_count() const noexcept;
        [[nodiscard]] size_type contraction_size() const noexcept;
        [[nodiscard]] size_type column_count() const noexcept;

        [[nodiscard]] const MatrixStrides &lhs_strides() const noexcept;
        [[nodiscard]] const MatrixStrides &rhs_strides() const noexcept;
        [[nodiscard]] const MatrixStrides &output_strides() const noexcept;

        // Calls function(batch_index, offsets) once for each broadcasted batch.
        // Offsets identify the first element of the logical matrix slices for
        // that batch. Matrix strides are constant across all batches.
        template <typename Function>
            requires std::invocable<Function &, size_type, const BatchOffsets &>
        void for_each_batch(Function &&function) const
        {
            if (batch_count_ == 0)
            {
                return;
            }

            std::vector<size_type> indices(dimensions_.size(), size_type{0});
            BatchOffsets offsets = initial_offsets_;

            for (size_type batch = 0; batch < batch_count_; ++batch)
            {
                std::invoke(function, batch, offsets);
                if (batch + 1 < batch_count_)
                {
                    advance(indices, dimensions_, offsets);
                }
            }
        }

    private:
        // represents batch dimensison
        struct Dimension final
        {
            size_type extent;
            BatchOffsets steps;
            BatchOffsets resets;
        };

        static void advance(
            std::span<size_type> indices,
            std::span<const Dimension> dimensions,
            BatchOffsets &offsets) noexcept
        {
            for (std::size_t i = dimensions.size(); i > 0; --i)
            {
                const std::size_t dimension_index = i - 1;
                const Dimension &dimension = dimensions[dimension_index];
                size_type &index = indices[dimension_index];

                ++index;
                if (index < dimension.extent)
                {
                    offsets.lhs += dimension.steps.lhs;
                    offsets.rhs += dimension.steps.rhs;
                    offsets.output += dimension.steps.output;
                    return;
                }

                index = 0;
                offsets.lhs -= dimension.resets.lhs;
                offsets.rhs -= dimension.resets.rhs;
                offsets.output -= dimension.resets.output;
            }
        }

        size_type batch_count_{1};
        size_type row_count_;
        size_type contraction_size_;
        size_type column_count_;

        MatrixStrides lhs_strides_;
        MatrixStrides rhs_strides_;
        MatrixStrides output_strides_;
        BatchOffsets initial_offsets_;
        std::vector<Dimension> dimensions_;
    };
}
