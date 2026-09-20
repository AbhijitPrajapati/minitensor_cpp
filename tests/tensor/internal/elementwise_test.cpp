#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "tensor/iteration/elementwise.hpp"
#include "tensor/storage/layout.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_elementwise_test()
    {
        using detail::ElementwisePlan;
        using detail::Layout;

        const std::array<Layout, 2> layouts{
            Layout({3, 1}, 1),
            Layout({0, -1}, 2)};
        const ElementwisePlan plan{Shape{2, 3}, layouts};
        expect(plan.numel() == 6, "an elementwise plan preserves the shape's element count");
        expect(plan.layout_count() == 2, "an elementwise plan preserves its layout count");

        struct Run final
        {
            Shape::size_type linear;
            std::array<Layout::offset_type, 2> offsets;
            std::array<Layout::stride_type, 2> strides;
            Shape::size_type size;

            bool operator==(const Run &) const = default;
        };

        std::vector<Run> visited_runs;
        std::vector<std::array<Layout::offset_type, 2>> visited_offsets;
        plan.for_each_run(
            [&visited_runs, &visited_offsets](Shape::size_type linear,
                                             std::span<const Layout::offset_type> offsets,
                                             std::span<const Layout::stride_type> strides,
                                             Shape::size_type size)
            {
                expect(offsets.size() == 2 && strides.size() == 2,
                       "an elementwise run reports one offset and stride per layout");
                visited_runs.push_back(Run{
                    linear,
                    {offsets[0], offsets[1]},
                    {strides[0], strides[1]},
                    size});

                Layout::offset_type first_offset = offsets[0];
                Layout::offset_type second_offset = offsets[1];
                for (Shape::size_type i = 0; i < size; ++i)
                {
                    expect(linear + i == visited_offsets.size(),
                           "elementwise runs cover increasing row-major linear indices");
                    visited_offsets.push_back({first_offset, second_offset});
                    first_offset += strides[0];
                    second_offset += strides[1];
                }
            });
        const std::vector<std::array<Layout::offset_type, 2>> expected_offsets{
            {1, 2}, {2, 1}, {3, 0}, {4, 2}, {5, 1}, {6, 0}};
        expect(visited_offsets == expected_offsets,
               "elementwise runs apply positive, zero, and negative strides across axis wraps");
        const std::vector<Run> expected_runs{
            {0, {1, 2}, {1, -1}, 3},
            {3, {4, 2}, {1, -1}, 3}};
        expect(visited_runs == expected_runs,
               "elementwise run iteration groups the innermost logical dimension");

        const std::array<Layout, 2> contiguous_layouts{
            Layout::contiguous(Shape{2, 3}),
            Layout::contiguous(Shape{2, 3}, 4)};
        const ElementwisePlan contiguous_plan{Shape{2, 3}, contiguous_layouts};
        std::size_t contiguous_run_calls = 0;
        contiguous_plan.for_each_run(
            [&contiguous_run_calls](Shape::size_type linear,
                                    std::span<const Layout::offset_type> offsets,
                                    std::span<const Layout::stride_type> strides,
                                    Shape::size_type size)
            {
                ++contiguous_run_calls;
                expect(linear == 0 && offsets[0] == 0 && offsets[1] == 4 &&
                           strides[0] == 1 && strides[1] == 1 && size == 6,
                       "contiguous dimensions are coalesced into one elementwise run");
            });
        expect(contiguous_run_calls == 1,
               "a fully contiguous elementwise plan emits one run");

        const std::array<Layout, 1> scalar_layout{Layout{}};
        const ElementwisePlan scalar_plan{Shape{}, scalar_layout};
        std::size_t scalar_run_calls = 0;
        scalar_plan.for_each_run(
            [&scalar_run_calls](Shape::size_type linear,
                                std::span<const Layout::offset_type> offsets,
                                std::span<const Layout::stride_type> strides,
                                Shape::size_type size)
            {
                ++scalar_run_calls;
                expect(linear == 0 && offsets[0] == 0 && strides[0] == 0 && size == 1,
                       "scalar run iteration reports one zero-stride element");
            });
        expect(scalar_run_calls == 1, "a scalar elementwise plan emits one run");

        const std::array<Layout, 1> empty_layout{Layout({4, 1}, 7)};
        const ElementwisePlan empty_plan{Shape{2, 0}, empty_layout};
        std::size_t empty_run_calls = 0;
        empty_plan.for_each_run(
            [&empty_run_calls](Shape::size_type,
                               std::span<const Layout::offset_type>,
                               std::span<const Layout::stride_type>,
                               Shape::size_type)
            {
                ++empty_run_calls;
            });
        expect(empty_plan.numel() == 0 && empty_run_calls == 0,
               "an empty elementwise plan emits no runs");

        expect_throws<std::invalid_argument>(
            []
            {
                (void)ElementwisePlan{Shape{2}, std::span<const Layout>{}};
            },
            "elementwise plan construction rejects an empty layout collection");
        expect_throws<std::invalid_argument>(
            []
            {
                const std::array<Layout, 1> wrong_rank{Layout({1})};
                (void)ElementwisePlan{Shape{2, 3}, wrong_rank};
            },
            "elementwise plan construction rejects a layout with the wrong rank");
    }
}
