#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/iteration/reduction.hpp"
#include "tensor/backend/cuda/register_kernels.hpp"
#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
TEST(CudaReductionPlanTest, PartitionsCoordinatesAndPreservesSignedStridesAndOffsets)
{
    const std::array<Shape::size_type, 2> axes{2, 0};
    const detail::cuda::ReductionPlan plan{Shape{2, 3, 4}, detail::Layout{{-12, 4, -1}, 15}, axes};
    EXPECT_EQ(plan.outer_shape(), Shape{3});
    EXPECT_EQ(plan.reduction_shape(), (Shape{2, 4}));
    EXPECT_EQ(plan.outer_layout(), (detail::Layout{{4}, 15}));
    EXPECT_EQ(plan.reduction_layout(), (detail::Layout{{-12, -1}}));
    EXPECT_EQ(plan.output_size(), 3U);
    EXPECT_EQ(plan.reduction_size(), 8U);
}

TEST(CudaReductionPlanTest, DistinguishesEmptyOutputsEmptyReductionsAndScalarCoordinates)
{
    const std::array<Shape::size_type, 1> axis{1};
    const Shape shape{2, 0, 3};
    const detail::cuda::ReductionPlan empty_reduction{shape, detail::Layout::contiguous(shape), axis};
    EXPECT_EQ(empty_reduction.output_size(), 6U);
    EXPECT_EQ(empty_reduction.reduction_size(), 0U);
    const std::array<Shape::size_type, 1> other_axis{2};
    const detail::cuda::ReductionPlan empty_output{shape, detail::Layout::contiguous(shape), other_axis};
    EXPECT_EQ(empty_output.output_size(), 0U);
    EXPECT_EQ(empty_output.reduction_size(), 3U);
    const detail::cuda::ReductionPlan scalar{Shape{}, detail::Layout{{}, 2}, {}};
    EXPECT_EQ(scalar.output_size(), 1U);
    EXPECT_EQ(scalar.reduction_size(), 1U);
    EXPECT_EQ(scalar.outer_layout().offset(), 2);
}

TEST(CudaReductionPlanTest, RejectsInvalidAxesAndLayoutRanks)
{
    const std::array<Shape::size_type, 2> duplicate{0, 0};
    const std::array<Shape::size_type, 1> outside{2};
    EXPECT_THROW((void)detail::cuda::ReductionPlan(Shape{2, 3}, detail::Layout{{3, 1}}, duplicate),
                 std::invalid_argument);
    EXPECT_THROW((void)detail::cuda::ReductionPlan(Shape{2, 3}, detail::Layout{{3, 1}}, outside),
                 std::invalid_argument);
    EXPECT_THROW((void)detail::cuda::ReductionPlan(Shape{2, 3}, detail::Layout{{1}}, {}), std::invalid_argument);
}

TEST(CudaReductionPlanTest, RejectsUnrepresentablePartitionSizes)
{
    const Shape shape{std::numeric_limits<Extent>::max(), 3, 0};
    const std::array<Shape::size_type, 1> axes{2};
    EXPECT_THROW((void)detail::cuda::ReductionPlan(shape, detail::Layout{{0, 0, 1}}, axes), std::overflow_error);
}

TEST(CudaReductionRegistrationTest, RegistersBothPrimitivesThroughTheBackendEntryPoint)
{
    detail::KernelRegistry registry;
    detail::cuda::register_kernels(registry);
    for (const auto id : {detail::SumPrimitive::ID, detail::LogSumExpPrimitive::ID})
    {
        EXPECT_TRUE(registry.contains({id, DeviceType::Cuda}));
        EXPECT_FALSE(registry.contains({id, DeviceType::Cpu}));
    }
}

class CudaReductionKernelTest : public CudaDeviceTest
{
  protected:
    template <typename PrimitiveType>
    void expect_reduction(const Shape &shape, const detail::Layout &layout, const std::vector<float> &values,
                          const std::vector<Axis> &axes, bool keep_dim, const Shape &output_shape,
                          const std::vector<float> &expected)
    {
        ASSERT_EQ(output_shape.numel(), expected.size());
        CudaKernelHarness harness;
        const auto device = harness.runtime().device();
        const detail::TensorSpec input_spec{shape, DType::Float32, device};
        const detail::TensorSpec output_spec{output_shape, DType::Float32, device};
        const auto input_buffer = harness.allocate<float>(values.size());
        harness.copy_from_host<float>(input_buffer, values);
        const detail::Materialization input_storage{input_buffer, layout};
        const auto output_buffer = harness.allocate<float>(expected.size() + 2);
        const std::vector<float> initial(expected.size() + 2, -42.0F);
        harness.copy_from_host<float>(output_buffer, initial);
        const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_shape, 1)};
        const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, input_storage}};
        const PrimitiveType primitive{axes, shape.rank(), keep_dim};
        const std::array<detail::TensorSpec, 1> specs{input_spec};
        ASSERT_EQ(primitive.infer(specs).shape, output_shape);
        harness.invoke(primitive, inputs, {output_spec, output_storage});

        const auto result = harness.copy_to_host<float>(output_buffer);
        EXPECT_EQ(result.front(), -42.0F);
        EXPECT_EQ(result.back(), -42.0F);
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            SCOPED_TRACE(i);
            if (std::isnan(expected[i]))
                EXPECT_TRUE(std::isnan(result[i + 1]));
            else if (std::isinf(expected[i]))
                EXPECT_EQ(result[i + 1], expected[i]);
            else
                EXPECT_NEAR(result[i + 1], expected[i], 1e-4F * std::max(1.0F, std::abs(expected[i])));
        }
        const auto unchanged = harness.copy_to_host<float>(input_buffer);
        ASSERT_EQ(unchanged.size(), values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            ASSERT_EQ(std::bit_cast<std::uint32_t>(unchanged[i]), std::bit_cast<std::uint32_t>(values[i]));
        }
    }
};

TEST_F(CudaReductionKernelTest, SumHandlesMultipleAxesNegativeStridesAndKeepDim)
{
    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 1.0F);
    expect_reduction<detail::SumPrimitive>(Shape{2, 3, 4}, detail::Layout{{-12, 4, -1}, 15}, values, {2, 0}, false,
                                           Shape{3}, {68, 100, 132});
    expect_reduction<detail::SumPrimitive>(Shape{2, 3, 4}, detail::Layout{{-12, 4, -1}, 15}, values, {-1, 0}, true,
                                           Shape{1, 3, 1}, {68, 100, 132});
}

TEST_F(CudaReductionKernelTest, HandlesBroadcastStridesAndInputOffsets)
{
    const auto log_two = std::log(2.0F);
    expect_reduction<detail::SumPrimitive>(Shape{2, 3}, detail::Layout{{0, 1}, 1}, {99, 1, 2, 3, 88}, {0}, false,
                                           Shape{3}, {2, 4, 6});
    expect_reduction<detail::LogSumExpPrimitive>(Shape{2, 3}, detail::Layout{{0, 1}, 1}, {99, 1, 2, 3, 88}, {0}, false,
                                                 Shape{3}, {1 + log_two, 2 + log_two, 3 + log_two});
}

TEST_F(CudaReductionKernelTest, ReducesSubBlockExactBlockAndPartialBlockExtents)
{
    for (const Extent count : {1, 17, 256, 1003})
    {
        SCOPED_TRACE(count);
        const std::vector<float> values(static_cast<std::size_t>(count), 1.0F);
        expect_reduction<detail::SumPrimitive>(Shape{count}, detail::Layout{{1}}, values, {0}, false, Shape{},
                                               {static_cast<float>(count)});
        expect_reduction<detail::LogSumExpPrimitive>(Shape{count}, detail::Layout{{1}}, values, {0}, false, Shape{},
                                                     {1.0F + std::log(static_cast<float>(count))});
    }
}

TEST_F(CudaReductionKernelTest, ReusesBlockScratchAcrossOutputElements)
{
    // Exceeds the launch grid cap, forcing blocks to process another output.
    constexpr Extent rows = 65537;
    std::vector<float> values(static_cast<std::size_t>(rows) * 3);
    std::vector<float> sums(rows);
    std::vector<float> log_sums(rows);
    const float shifted_log_sum = std::log(1.0F + std::exp(-1.0F) + std::exp(-2.0F));
    for (std::size_t row = 0; row < sums.size(); ++row)
    {
        const auto base = static_cast<float>(row % 7);
        for (std::size_t k = 0; k < 3; ++k)
            values[row * 3 + k] = base + static_cast<float>(k);
        sums[row] = 3 * base + 3;
        log_sums[row] = base + 2 + shifted_log_sum;
    }
    expect_reduction<detail::SumPrimitive>(Shape{rows, 3}, detail::Layout{{3, 1}}, values, {1}, false, Shape{rows},
                                           sums);
    expect_reduction<detail::LogSumExpPrimitive>(Shape{rows, 3}, detail::Layout{{3, 1}}, values, {1}, false,
                                                 Shape{rows}, log_sums);
}

TEST_F(CudaReductionKernelTest, LogSumExpIsStableAndPropagatesNaNsBeforeInfinities)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float shift = std::log(1.0F + std::exp(-1.0F) + std::exp(-2.0F));
    expect_reduction<detail::LogSumExpPrimitive>(
        Shape{5, 3}, detail::Layout{{3, 1}},
        {1000, 1001, 999, -1000, -1001, -999, -inf, -inf, -inf, inf, 0, -inf, nan, inf, -inf}, {1}, false, Shape{5},
        {1001 + shift, -999 + shift, -inf, inf, nan});
}

TEST_F(CudaReductionKernelTest, SumPreservesNonfiniteSemantics)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    expect_reduction<detail::SumPrimitive>(Shape{4, 2}, detail::Layout{{2, 1}}, {nan, 1, inf, 2, -inf, 2, inf, -inf},
                                           {1}, false, Shape{4}, {nan, inf, -inf, nan});
}

TEST_F(CudaReductionKernelTest, SupportsScalarsAndNoReducedAxes)
{
    expect_reduction<detail::SumPrimitive>(Shape{}, detail::Layout{{}, 1}, {99, 7, 88}, {}, false, Shape{}, {7});
    expect_reduction<detail::LogSumExpPrimitive>(Shape{}, detail::Layout{{}, 1}, {99, 7, 88}, {}, true, Shape{}, {7});
    expect_reduction<detail::SumPrimitive>(Shape{2, 2}, detail::Layout{{1, 2}}, {1, 3, 2, 4}, {}, true, Shape{2, 2},
                                           {1, 2, 3, 4});
    expect_reduction<detail::LogSumExpPrimitive>(Shape{2, 2}, detail::Layout{{1, 2}}, {1, 3, 2, 4}, {}, false,
                                                 Shape{2, 2}, {1, 2, 3, 4});
}

TEST_F(CudaReductionKernelTest, EmptyReductionsWriteIdentitiesAndEmptyOutputsRemainUntouched)
{
    const Shape shape{2, 0, 3};
    const float negative_inf = -std::numeric_limits<float>::infinity();
    expect_reduction<detail::SumPrimitive>(shape, detail::Layout::contiguous(shape), {}, {1}, false, Shape{2, 3},
                                           std::vector<float>(6, 0));
    expect_reduction<detail::LogSumExpPrimitive>(shape, detail::Layout::contiguous(shape), {}, {1}, true,
                                                 Shape{2, 1, 3}, std::vector<float>(6, negative_inf));
    expect_reduction<detail::SumPrimitive>(shape, detail::Layout::contiguous(shape), {}, {2}, false, Shape{2, 0}, {});
    expect_reduction<detail::LogSumExpPrimitive>(shape, detail::Layout::contiguous(shape), {}, {2}, false, Shape{2, 0},
                                                 {});
    expect_reduction<detail::SumPrimitive>(Shape{0}, detail::Layout{{1}}, {}, {0}, false, Shape{}, {0});
    expect_reduction<detail::LogSumExpPrimitive>(Shape{0}, detail::Layout{{1}}, {}, {0}, false, Shape{},
                                                 {negative_inf});
}
} // namespace minitensor::test
