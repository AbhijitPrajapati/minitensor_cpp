#include <minitensor/types.hpp>

#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/iteration/matmul.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
TEST(CpuMatmulPlanTest, ProducesBroadcastBatchOffsetsAndMatrixGeometry)
{
    const Shape lhs_shape{2, 1, 1, 2};
    const Shape rhs_shape{3, 2, 1};
    const Shape output_shape{2, 3, 1, 1};
    const detail::cpu::MatmulPlan plan{lhs_shape,    detail::Layout::contiguous(lhs_shape),
                                       rhs_shape,    detail::Layout::contiguous(rhs_shape),
                                       output_shape, detail::Layout::contiguous(output_shape)};
    std::vector<detail::cpu::MatmulPlan::BatchOffsets> offsets;

    plan.for_each_batch(
        [&](Shape::size_type, const detail::cpu::MatmulPlan::BatchOffsets &batch) { offsets.push_back(batch); });

    EXPECT_EQ(plan.batch_count(), 6);
    EXPECT_EQ(plan.row_count(), 1);
    EXPECT_EQ(plan.contraction_size(), 2);
    EXPECT_EQ(plan.column_count(), 1);
    EXPECT_EQ(plan.lhs_strides(), (detail::cpu::MatmulPlan::MatrixStrides{2, 1}));
    EXPECT_EQ(plan.rhs_strides(), (detail::cpu::MatmulPlan::MatrixStrides{1, 1}));
    EXPECT_EQ(plan.output_strides(), (detail::cpu::MatmulPlan::MatrixStrides{1, 1}));
    EXPECT_EQ(offsets, (std::vector<detail::cpu::MatmulPlan::BatchOffsets>{
                           {0, 0, 0}, {0, 2, 1}, {0, 4, 2}, {2, 0, 3}, {2, 2, 4}, {2, 4, 5}}));
}
} // namespace minitensor::test
