#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
TEST(SumTest, InfersReducedShapes)
{
    const Tensor input = full(Shape{2, 3, 4}, 1.0F);

    EXPECT_EQ(sum(input, {-1}).shape(), (Shape{2, 3}));
    EXPECT_EQ(sum(input, {2}, true).shape(), (Shape{2, 3, 1}));
    EXPECT_EQ(sum(input, {2, 0}).shape(), (Shape{3}));
    EXPECT_EQ(sum(input).shape(), Shape{});
    EXPECT_EQ(sum(input, true).shape(), (Shape{1, 1, 1}));
    EXPECT_EQ(sum(input, std::span<const Axis>{}).shape(), input.shape());
    EXPECT_TRUE(sum(full(Shape{}, 1.0F)).shape().is_scalar());
}

TEST(SumTest, ReducesContiguousAndStridedInputs)
{
    const std::array<float, 6> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const Tensor matrix = from_data(values, Shape{2, 3});

    const std::array<float, 2> row_sums{6.0F, 15.0F};
    expect_tensor_eq(sum(matrix, {-1}), row_sums);
    expect_tensor_eq(sum(matrix, {-1}, true), row_sums);
    EXPECT_FLOAT_EQ(item(sum(matrix)), 21.0F);

    const Tensor transposed = transpose(matrix);
    const std::array<float, 3> transposed_rows{5.0F, 7.0F, 9.0F};
    expect_tensor_eq(sum(transposed, {1}), transposed_rows);
}

TEST(SumTest, ReducesBroadcastedAndNonAdjacentDimensions)
{
    const Tensor row = from_data({7.0F, 8.0F, 9.0F}, Shape{1, 3});
    const Tensor broadcast = broadcast_to(row, Shape{2, 3});
    const std::array<float, 3> broadcast_sums{14.0F, 16.0F, 18.0F};
    expect_tensor_eq(sum(broadcast, {0}), broadcast_sums);

    const std::array<float, 12> volume_values{1.0F, 2.0F, 3.0F, 4.0F,  5.0F,  6.0F,
                                              7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    const Tensor volume = from_data(volume_values, Shape{2, 3, 2});
    const std::array<float, 3> expected{18.0F, 26.0F, 34.0F};
    expect_tensor_eq(sum(volume, {0, 2}), expected);
}

TEST(SumTest, UsesTheAdditiveIdentityForEmptyReductions)
{
    const Tensor empty = full(Shape{2, 0, 3}, 7.0F);
    const std::array<float, 6> expected{};
    const Tensor result = sum(empty, {1});

    EXPECT_EQ(result.shape(), (Shape{2, 3}));
    expect_tensor_eq(result, expected);
}

TEST(SumTest, ValidatesAxes)
{
    const Tensor matrix = full(Shape{2, 3}, 1.0F);
    EXPECT_THROW((void)sum(matrix, {1, -1}), std::invalid_argument);
    EXPECT_THROW((void)sum(matrix, {2}), std::out_of_range);
    EXPECT_THROW((void)sum(matrix, {0, 1, 0}), std::invalid_argument);
}

TEST(SumTest, RejectsIntegralInputs)
{
    const Tensor input = from_data<std::int64_t>({1, 2}, Shape{2});
    EXPECT_THROW((void)sum(input), std::invalid_argument);
}

TEST(MeanTest, ReducesContiguousAndStridedInputs)
{
    const Tensor matrix = from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3});

    const std::array<float, 2> row_means{2.0F, 5.0F};
    expect_tensor_eq(mean(matrix, {-1}), row_means);
    const Tensor kept = mean(matrix, {-1}, true);
    EXPECT_EQ(kept.shape(), (Shape{2, 1}));
    expect_tensor_eq(kept, row_means);
    EXPECT_FLOAT_EQ(item(mean(matrix)), 3.5F);

    const std::array<float, 3> transposed_rows{2.5F, 3.5F, 4.5F};
    expect_tensor_eq(mean(transpose(matrix), {1}), transposed_rows);
}

TEST(LogSumExpTest, ReducesStridedInputsWithoutOverflow)
{
    const Tensor matrix = from_data({1000.0F, -1000.0F, 0.0F, 1000.0F, -1000.0F, 0.0F}, Shape{2, 3});
    const Tensor transposed = transpose(matrix);
    const Tensor result = logsumexp(transposed, {-1}, true);

    EXPECT_EQ(result.shape(), (Shape{3, 1}));
    const std::array<float, 3> expected{1000.0F + std::log(2.0F), -1000.0F + std::log(2.0F), std::log(2.0F)};
    expect_tensor_near(result, expected, 1.0E-4F, 0.0F);
    EXPECT_NEAR(item(logsumexp(matrix)), 1000.0F + std::log(2.0F), 1.0E-4F);
}

TEST(LogSumExpTest, HandlesEmptyAndNonFiniteReductions)
{
    const Tensor empty = full(Shape{2, 0, 3}, 0.0F);
    const std::vector<float> empty_result = to_vector(logsumexp(empty, {1}));
    ASSERT_EQ(empty_result.size(), 6);
    for (const float value : empty_result)
    {
        EXPECT_TRUE(std::isinf(value));
        EXPECT_TRUE(std::signbit(value));
    }

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor special = from_data({-infinity, -infinity, infinity, 0.0F, nan, 1.0F}, Shape{3, 2});
    const std::vector<float> special_result = to_vector(logsumexp(special, {1}));

    ASSERT_EQ(special_result.size(), 3);
    EXPECT_TRUE(std::isinf(special_result[0]));
    EXPECT_TRUE(std::signbit(special_result[0]));
    EXPECT_EQ(special_result[1], infinity);
    EXPECT_TRUE(std::isnan(special_result[2]));
}

TEST(LogSumExpTest, ValidatesAxesAndDType)
{
    const Tensor matrix = full(Shape{2, 3}, 1.0F);
    EXPECT_THROW((void)logsumexp(matrix, {1, -1}), std::invalid_argument);
    EXPECT_THROW((void)logsumexp(matrix, {2}), std::out_of_range);

    const Tensor integral = from_data<std::int64_t>({1, 2}, Shape{2});
    EXPECT_THROW((void)logsumexp(integral), std::invalid_argument);
}
} // namespace minitensor::test
