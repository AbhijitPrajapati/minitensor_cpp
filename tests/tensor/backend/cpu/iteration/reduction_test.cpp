#include <minitensor/types.hpp>

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/iteration/reduction.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
TEST(CpuReductionPlanTest, SeparatesOutputsFromReductionRuns)
{
    const std::array<Shape::size_type, 1> axes{1};
    const detail::cpu::ReductionPlan plan{Shape{2, 3}, detail::Layout{{-3, 1}, 3}, axes};
    std::vector<Shape::size_type> output_indices;
    std::vector<detail::cpu::ReductionPlan::Run> runs;

    plan.for_each_output([&](Shape::size_type output_index, const detail::cpu::ReductionPlan::ReductionRange &range) {
        EXPECT_EQ(range.size(), 3);
        output_indices.push_back(output_index);
        range.for_each_run([&](const detail::cpu::ReductionPlan::Run &run) { runs.push_back(run); });
    });

    ASSERT_EQ(runs.size(), 2);
    EXPECT_EQ(output_indices, (std::vector<Shape::size_type>{0, 1}));
    EXPECT_EQ(runs[0].offset, 3);
    EXPECT_EQ(runs[1].offset, 0);
    for (const detail::cpu::ReductionPlan::Run &run : runs)
    {
        EXPECT_EQ(run.stride, 1);
        EXPECT_EQ(run.size, 3);
    }
}
} // namespace minitensor::test
