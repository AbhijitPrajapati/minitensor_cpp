#include <minitensor/types.hpp>

#include <array>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/iteration/elementwise.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
TEST(CpuElementwisePlanTest, ProducesRunsForStridedAndBroadcastLayouts)
{
    const Shape shape{2, 3};
    const std::array<detail::Layout, 2> layouts{detail::Layout{{-3, 1}, 3}, detail::Layout{{0, 1}}};
    const detail::cpu::ElementwisePlan plan{shape, layouts};
    std::vector<Shape::size_type> linear_indices;
    std::vector<std::vector<detail::Layout::offset_type>> offsets;
    std::vector<std::vector<detail::Layout::stride_type>> strides;
    std::vector<Shape::size_type> run_sizes;

    plan.for_each_run([&](Shape::size_type linear, std::span<const detail::Layout::offset_type> run_offsets,
                          std::span<const detail::Layout::stride_type> run_strides, Shape::size_type run_size) {
        linear_indices.push_back(linear);
        offsets.emplace_back(run_offsets.begin(), run_offsets.end());
        strides.emplace_back(run_strides.begin(), run_strides.end());
        run_sizes.push_back(run_size);
    });

    EXPECT_EQ(linear_indices, (std::vector<Shape::size_type>{0, 3}));
    EXPECT_EQ(offsets, (std::vector<std::vector<detail::Layout::offset_type>>{{3, 0}, {0, 0}}));
    EXPECT_EQ(strides, (std::vector<std::vector<detail::Layout::stride_type>>{{1, 1}, {1, 1}}));
    EXPECT_EQ(run_sizes, (std::vector<Shape::size_type>{3, 3}));
}
} // namespace minitensor::test
