#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "tensor/iteration/reduction.hpp"
#include "tensor/storage/layout.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_reduction_test()
    {
        using detail::Layout;
        using detail::ReductionPlan;

        using Visit = std::pair<Shape::size_type, Layout::offset_type>;

        constexpr std::array<Shape::size_type, 2> reduced_axes{0, 2};
        const ReductionPlan plan{
            Shape{2, 3, 2},
            Layout({2, 6, -1}, 1),
            Shape{3},
            reduced_axes};
        expect(plan.input_numel() == 12,
               "a reduction plan preserves the input element count");
        expect(plan.output_numel() == 3,
               "a reduction plan preserves the output element count");

        std::vector<Visit> visits;
        plan.for_each(
            [&visits](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                visits.emplace_back(output_linear, input_offset);
            });

        const std::vector<Visit> expected_visits{
            {0, 1}, {0, 0}, {0, 3}, {0, 2},
            {1, 7}, {1, 6}, {1, 9}, {1, 8},
            {2, 13}, {2, 12}, {2, 15}, {2, 14}};
        expect(visits == expected_visits,
               "reduction iteration groups non-adjacent reduced axes by output and applies offsets and signed strides");

        constexpr std::array<Shape::size_type, 1> leading_axis{0};
        const ReductionPlan broadcast_plan{
            Shape{2, 3},
            Layout({0, 1}, 4),
            Shape{3},
            leading_axis};
        std::vector<Visit> broadcast_visits;
        broadcast_plan.for_each(
            [&broadcast_visits](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                broadcast_visits.emplace_back(output_linear, input_offset);
            });
        const std::vector<Visit> expected_broadcast_visits{
            {0, 4}, {0, 4}, {1, 5}, {1, 5}, {2, 6}, {2, 6}};
        expect(broadcast_visits == expected_broadcast_visits,
               "reduction iteration honors zero strides in broadcasted input layouts");

        const ReductionPlan no_axes_plan{
            Shape{2, 2},
            Layout({1, 3}, 2),
            Shape{2, 2},
            std::span<const Shape::size_type>{}};
        std::vector<Visit> no_axes_visits;
        no_axes_plan.for_each(
            [&no_axes_visits](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                no_axes_visits.emplace_back(output_linear, input_offset);
            });
        const std::vector<Visit> expected_no_axes_visits{
            {0, 2}, {1, 5}, {2, 3}, {3, 6}};
        expect(no_axes_visits == expected_no_axes_visits,
               "a reduction over no axes visits each logical input once");

        constexpr std::array<Shape::size_type, 2> all_axes{0, 1};
        const ReductionPlan all_axes_plan{
            Shape{2, 2},
            Layout::contiguous(Shape{2, 2}, 1),
            Shape{},
            all_axes};
        std::vector<Visit> all_axes_visits;
        all_axes_plan.for_each(
            [&all_axes_visits](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                all_axes_visits.emplace_back(output_linear, input_offset);
            });
        const std::vector<Visit> expected_all_axes_visits{
            {0, 1}, {0, 2}, {0, 3}, {0, 4}};
        expect(all_axes_visits == expected_all_axes_visits,
               "a reduction over every axis associates every input with the scalar output");

        const ReductionPlan scalar_plan{
            Shape{},
            Layout(std::vector<Layout::stride_type>{}, 5),
            Shape{},
            std::span<const Shape::size_type>{}};
        std::vector<Visit> scalar_visits;
        scalar_plan.for_each(
            [&scalar_visits](Shape::size_type output_linear, Layout::offset_type input_offset)
            {
                scalar_visits.emplace_back(output_linear, input_offset);
            });
        expect(scalar_visits == std::vector<Visit>{{0, 5}},
               "scalar reduction iteration invokes its callback once at the layout offset");

        constexpr std::array<Shape::size_type, 1> empty_axis{1};
        const Shape empty_input_shape{2, 0, 3};
        const ReductionPlan empty_reduction_plan{
            empty_input_shape,
            Layout::contiguous(empty_input_shape),
            Shape{2, 3},
            empty_axis};
        std::size_t empty_reduction_calls = 0;
        empty_reduction_plan.for_each(
            [&empty_reduction_calls](Shape::size_type, Layout::offset_type)
            {
                ++empty_reduction_calls;
            });
        expect(empty_reduction_plan.input_numel() == 0 &&
                   empty_reduction_plan.output_numel() == 6 &&
                   empty_reduction_calls == 0,
               "reducing a zero-length axis emits no input visits for its nonempty output");

        constexpr std::array<Shape::size_type, 1> nonempty_axis{0};
        const ReductionPlan empty_output_plan{
            empty_input_shape,
            Layout::contiguous(empty_input_shape),
            Shape{0, 3},
            nonempty_axis};
        std::size_t empty_output_calls = 0;
        empty_output_plan.for_each(
            [&empty_output_calls](Shape::size_type, Layout::offset_type)
            {
                ++empty_output_calls;
            });
        expect(empty_output_plan.output_numel() == 0 && empty_output_calls == 0,
               "a reduction with an empty output performs no callbacks");
    }
}
