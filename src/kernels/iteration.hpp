#pragma once

#include "core/layout.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

namespace minitensor::detail::kernel
{

    // Advances validated row-major logical coordinates by one element.
    void advance_coordinates(const Shape &shape, Coordinates &coordinates) noexcept;

    // Coordinates a single logical elementwise traversal across one output
    // and any number of broadcast-compatible inputs.
    class ElementwiseIterator final
    {
    public:
        ElementwiseIterator(const Layout &output, std::initializer_list<Layout> inputs);

        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] Index numel() const noexcept;
        [[nodiscard]] std::size_t input_count() const noexcept;
        [[nodiscard]] const Layout &output_layout() const noexcept;
        [[nodiscard]] const Layout &input_layout(std::size_t index) const;

        template <typename Visitor>
        void for_each(Visitor &&visitor) const
        {
            std::vector<Index> offsets(layouts_.size());
            Coordinates coordinates(shape_.size(), 0);
            for (Index linear = 0; linear < numel_; ++linear)
            {
                for (std::size_t index = 0; index < layouts_.size(); ++index)
                {
                    offsets[index] = layouts_[index].offset_from_coordinates(coordinates);
                }

                const std::span<const Index> input_offsets(
                    offsets.data() + 1,
                    offsets.size() - 1);
                std::invoke(visitor, offsets.front(), input_offsets);
                advance_coordinates(shape_, coordinates);
            }
        }

    private:
        Shape shape_;
        Index numel_{0};
        // The output layout is first; all following layouts are inputs
        // normalized to the common logical shape.
        std::vector<Layout> layouts_;
    };

    // Coordinates each logical input element with the output element into
    // which it reduces. The actual output layout is normalized to input rank
    // internally, making removed and retained reduction dimensions uniform.
    class ReductionIterator final
    {
    public:
        ReductionIterator(
            const Layout &input,
            const Layout &output,
            std::optional<Index> dimension,
            bool keepdim);

        // Builds the inverse-broadcast mapping used when accumulating an
        // expanded value into a smaller target shape.
        [[nodiscard]] static ReductionIterator to_shape(
            const Layout &input,
            const Layout &output);

        [[nodiscard]] const Shape &input_shape() const noexcept;
        [[nodiscard]] const Shape &output_shape() const noexcept;
        [[nodiscard]] Index numel() const noexcept;
        [[nodiscard]] std::span<const Index> reduced_axes() const noexcept;
        [[nodiscard]] bool keepdim() const noexcept;
        [[nodiscard]] const Layout &input_layout() const noexcept;
        [[nodiscard]] const Layout &output_layout() const noexcept;

        template <typename Visitor>
        void for_each(Visitor &&visitor) const
        {
            Coordinates input_coordinates(input_shape_.size(), 0);
            Coordinates output_coordinates(input_shape_.size(), 0);
            for (Index linear = 0; linear < numel_; ++linear)
            {
                for (std::size_t dimension = 0; dimension < input_shape_.size(); ++dimension)
                {
                    output_coordinates[dimension] = reduced_axis_mask_[dimension]
                                                        ? 0
                                                        : input_coordinates[dimension];
                }

                const auto input_offset =
                    input_layout_.offset_from_coordinates(input_coordinates);
                const auto output_offset =
                    normalized_output_layout_.offset_from_coordinates(output_coordinates);
                std::invoke(visitor, output_offset, input_offset);
                advance_coordinates(input_shape_, input_coordinates);
            }
        }

    private:
        struct Configuration final
        {
            std::vector<Index> reduced_axes;
            bool keepdim;
            Layout normalized_output_layout;
        };

        ReductionIterator(
            const Layout &input,
            const Layout &output,
            Configuration configuration);

        [[nodiscard]] static Configuration configure_standard(
            const Layout &input,
            const Layout &output,
            std::optional<Index> dimension,
            bool keepdim);
        [[nodiscard]] static Configuration configure_to_shape(
            const Layout &input,
            const Layout &output);

        Shape input_shape_;
        Shape output_shape_;
        Index numel_{0};
        std::vector<Index> reduced_axes_;
        bool keepdim_{false};
        std::vector<bool> reduced_axis_mask_;
        Layout input_layout_;
        Layout output_layout_;
        Layout normalized_output_layout_;
    };

    // Owns the rank-2 contraction geometry used by matrix multiplication.
    // Numerical multiplication and accumulation remain the kernel's concern.
    class MatrixMultiplyPlan final
    {
    public:
        MatrixMultiplyPlan(
            const Layout &lhs,
            const Layout &rhs,
            const Layout &output);

        [[nodiscard]] Index rows() const noexcept;
        [[nodiscard]] Index inner_size() const noexcept;
        [[nodiscard]] Index columns() const noexcept;
        [[nodiscard]] const Layout &lhs_layout() const noexcept;
        [[nodiscard]] const Layout &rhs_layout() const noexcept;
        [[nodiscard]] const Layout &output_layout() const noexcept;

        template <typename Visitor>
        void for_each(Visitor &&visitor) const
        {
            for (Index row = 0; row < rows_; ++row)
            {
                for (Index column = 0; column < columns_; ++column)
                {
                    const std::array<Index, 2> output_coordinates{row, column};
                    const auto output_offset =
                        output_layout_.offset_from_coordinates(output_coordinates);
                    for (Index inner = 0; inner < inner_size_; ++inner)
                    {
                        const std::array<Index, 2> lhs_coordinates{row, inner};
                        const std::array<Index, 2> rhs_coordinates{inner, column};
                        std::invoke(
                            visitor,
                            output_offset,
                            lhs_layout_.offset_from_coordinates(lhs_coordinates),
                            rhs_layout_.offset_from_coordinates(rhs_coordinates));
                    }
                }
            }
        }

    private:
        Layout lhs_layout_;
        Layout rhs_layout_;
        Layout output_layout_;
        Index rows_{0};
        Index inner_size_{0};
        Index columns_{0};
    };

} // namespace minitensor::detail::kernel
