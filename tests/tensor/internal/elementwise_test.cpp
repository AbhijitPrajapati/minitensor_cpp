#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <limits>
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

        std::vector<std::array<Layout::offset_type, 2>> visited_offsets;
        plan.for_each(
            [&visited_offsets](Shape::size_type linear,
                               std::span<const Layout::offset_type> offsets)
            {
                expect(linear == visited_offsets.size(),
                       "elementwise iteration emits increasing row-major linear indices");
                expect(offsets.size() == 2,
                       "elementwise iteration emits one offset for each layout");
                visited_offsets.push_back({offsets[0], offsets[1]});
            });

        const std::vector<std::array<Layout::offset_type, 2>> expected_offsets{
            {1, 2}, {2, 1}, {3, 0}, {4, 2}, {5, 1}, {6, 0}};
        expect(visited_offsets == expected_offsets,
               "elementwise iteration applies positive, zero, and negative strides across axis wraps");

        const std::array<Layout, 1> scalar_layout{Layout{}};
        const ElementwisePlan scalar_plan{Shape{}, scalar_layout};
        std::size_t scalar_calls = 0;
        scalar_plan.for_each(
            [&scalar_calls](Shape::size_type linear,
                            std::span<const Layout::offset_type> offsets)
            {
                ++scalar_calls;
                expect(linear == 0 && offsets.size() == 1 && offsets[0] == 0,
                       "scalar iteration reports its sole logical element and layout offset");
            });
        expect(scalar_calls == 1, "a scalar elementwise plan invokes its function once");

        const std::array<Layout, 1> empty_layout{Layout({4, 1}, 7)};
        const ElementwisePlan empty_plan{Shape{2, 0}, empty_layout};
        std::size_t empty_calls = 0;
        empty_plan.for_each(
            [&empty_calls](Shape::size_type, std::span<const Layout::offset_type>)
            {
                ++empty_calls;
            });
        expect(empty_plan.numel() == 0 && empty_calls == 0,
               "an empty elementwise plan performs no callbacks");

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
        expect_throws<std::overflow_error>(
            []
            {
                const std::array<Layout, 1> overflowing{
                    Layout({std::numeric_limits<Layout::stride_type>::max()})};
                (void)ElementwisePlan{Shape{3}, overflowing};
            },
            "elementwise plan construction rejects an overflowing axis reset");
    }
}
