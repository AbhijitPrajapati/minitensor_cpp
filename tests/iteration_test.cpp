#include "kernels/iteration.hpp"

#include <array>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using minitensor::Index;
    using minitensor::Shape;
    using minitensor::Strides;
    using minitensor::detail::ElementwiseIterator;
    using minitensor::detail::Layout;
    using minitensor::detail::MatrixMultiplyPlan;
    using minitensor::detail::ReductionIterator;

    int failures = 0;

    void expect(const bool condition, const std::string &message)
    {
        if (!condition)
        {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Exception>
    void expect_throws(const std::function<void()> &action, const std::string &message)
    {
        try
        {
            action();
            expect(false, message + " (no exception was thrown)");
        }
        catch (const Exception &)
        {
        }
        catch (...)
        {
            expect(false, message + " (wrong exception type)");
        }
    }

} // namespace

int main()
{
    const auto output = Layout::contiguous(Shape{2, 3}, 5);
    const auto input = Layout::contiguous(Shape{3}, 2);
    const ElementwiseIterator iterator(output, {input});

    expect(iterator.shape() == Shape{2, 3}, "iterator owns the common logical shape");
    expect(iterator.numel() == 6, "iterator owns the logical element count");
    expect(iterator.input_count() == 1, "iterator records each input layout");
    expect(
        iterator.output_layout().offset() == 5,
        "iterator preserves the output layout");
    expect(
        iterator.input_layout(0).shape() == Shape{2, 3} &&
            iterator.input_layout(0).strides() == Strides{0, 1},
        "iterator normalizes a broadcast input layout");

    std::vector<std::pair<Index, Index>> visited_offsets;
    iterator.for_each(
        [&](const Index output_offset, const std::span<const Index> input_offsets)
        { visited_offsets.emplace_back(output_offset, input_offsets[0]); });
    expect(
        visited_offsets == std::vector<std::pair<Index, Index>>{
                               {5, 2},
                               {6, 3},
                               {7, 4},
                               {8, 2},
                               {9, 3},
                               {10, 4}},
        "iterator emits coordinated physical offsets in logical order");

    Index fill_visits = 0;
    const ElementwiseIterator fill_iterator(output, {});
    fill_iterator.for_each(
        [&](const Index, const std::span<const Index> input_offsets)
        {
            expect(input_offsets.empty(), "zero-input iteration exposes no input offsets");
            ++fill_visits;
        });
    expect(fill_visits == 6, "zero-input iteration supports fill kernels");

    expect_throws<std::logic_error>(
        [&]
        {
            static_cast<void>(ElementwiseIterator(
                Layout::contiguous(Shape{2, 2}),
                {Layout::contiguous(Shape{2, 3})}));
        },
        "iterator rejects an input that cannot broadcast to the output shape");

    const auto reduction_input = Layout::contiguous(Shape{2, 3}, 10);
    const auto reduction_output = Layout::contiguous(Shape{2}, 4);
    const ReductionIterator reduction_iterator(
        reduction_input,
        reduction_output,
        std::optional<Index>{1},
        false);
    expect(
        reduction_iterator.input_shape() == Shape{2, 3} &&
            reduction_iterator.output_shape() == Shape{2},
        "reduction iterator owns its input and output shapes");
    expect(reduction_iterator.numel() == 6, "reduction iterator owns input iteration size");
    expect(
        std::vector<Index>(
            reduction_iterator.reduced_axes().begin(),
            reduction_iterator.reduced_axes().end()) == std::vector<Index>{1},
        "reduction iterator owns the reduced axes");
    expect(!reduction_iterator.keepdim(), "reduction iterator records removed dimensions");

    std::vector<std::pair<Index, Index>> reduction_offsets;
    reduction_iterator.for_each(
        [&](const Index output_offset, const Index input_offset)
        { reduction_offsets.emplace_back(output_offset, input_offset); });
    expect(
        reduction_offsets == std::vector<std::pair<Index, Index>>{
                                 {4, 10},
                                 {4, 11},
                                 {4, 12},
                                 {5, 13},
                                 {5, 14},
                                 {5, 15}},
        "reduction iterator maps each input element to its output bucket");

    const auto transposed_input = Layout(Shape{3, 2}, Strides{1, 3}, 2);
    const auto kept_output = Layout::contiguous(Shape{3, 1}, 8);
    const ReductionIterator kept_iterator(
        transposed_input,
        kept_output,
        std::optional<Index>{1},
        true);
    std::vector<std::pair<Index, Index>> kept_offsets;
    kept_iterator.for_each(
        [&](const Index output_offset, const Index input_offset)
        { kept_offsets.emplace_back(output_offset, input_offset); });
    expect(kept_iterator.keepdim(), "reduction iterator records retained dimensions");
    expect(
        kept_offsets == std::vector<std::pair<Index, Index>>{
                            {8, 2},
                            {8, 5},
                            {9, 3},
                            {9, 6},
                            {10, 4},
                            {10, 7}},
        "reduction iterator honors strided input and keepdim output layouts");

    const auto to_shape_iterator = ReductionIterator::to_shape(
        Layout::contiguous(Shape{2, 4, 3}, 0),
        Layout::contiguous(Shape{1, 3}, 20));
    expect(
        std::vector<Index>(
            to_shape_iterator.reduced_axes().begin(),
            to_shape_iterator.reduced_axes().end()) == std::vector<Index>{0, 1},
        "inverse-broadcast reduction identifies leading and singleton axes");
    std::vector<std::pair<Index, Index>> to_shape_offsets;
    to_shape_iterator.for_each(
        [&](const Index output_offset, const Index input_offset)
        { to_shape_offsets.emplace_back(output_offset, input_offset); });
    expect(
        to_shape_offsets.front() == std::pair<Index, Index>{20, 0} &&
            to_shape_offsets[3] == std::pair<Index, Index>{20, 3} &&
            to_shape_offsets[12] == std::pair<Index, Index>{20, 12} &&
            to_shape_offsets.back() == std::pair<Index, Index>{22, 23},
        "inverse-broadcast reduction maps expanded axes into target buckets");

    expect_throws<std::logic_error>(
        [&]
        {
            static_cast<void>(ReductionIterator(
                Layout::contiguous(Shape{2, 3}),
                Layout::contiguous(Shape{3}),
                std::optional<Index>{1},
                false));
        },
        "reduction iterator rejects an output shape inconsistent with its axis");

    const MatrixMultiplyPlan matrix_plan(
        Layout::contiguous(Shape{2, 3}, 2),
        Layout(Shape{3, 2}, Strides{1, 3}, 10),
        Layout::contiguous(Shape{2, 2}, 20));
    expect(
        matrix_plan.rows() == 2 &&
            matrix_plan.inner_size() == 3 &&
            matrix_plan.columns() == 2,
        "matrix multiplication plan owns its contraction dimensions");
    expect(
        matrix_plan.lhs_layout().offset() == 2 &&
            matrix_plan.rhs_layout().offset() == 10 &&
            matrix_plan.output_layout().offset() == 20,
        "matrix multiplication plan owns all operand layouts");

    std::vector<std::array<Index, 3>> matrix_offsets;
    matrix_plan.for_each(
        [&](const Index output_offset, const Index lhs_offset, const Index rhs_offset)
        { matrix_offsets.push_back({output_offset, lhs_offset, rhs_offset}); });
    expect(matrix_offsets.size() == 12, "matrix plan visits every contraction product");
    expect(
        matrix_offsets.front() == std::array<Index, 3>{20, 2, 10} &&
            matrix_offsets[2] == std::array<Index, 3>{20, 4, 12} &&
            matrix_offsets[3] == std::array<Index, 3>{21, 2, 13} &&
            matrix_offsets.back() == std::array<Index, 3>{23, 7, 15},
        "matrix plan coordinates strided operands with each output bucket");

    expect_throws<std::logic_error>(
        [&]
        {
            static_cast<void>(MatrixMultiplyPlan(
                Layout::contiguous(Shape{2, 3}),
                Layout::contiguous(Shape{3, 2}),
                Layout::contiguous(Shape{2, 3})));
        },
        "matrix plan rejects an inconsistent output shape");

    if (failures != 0)
    {
        std::cerr << failures << " iteration test(s) failed\n";
        return 1;
    }
    std::cout << "All iteration tests passed\n";
    return 0;
}
