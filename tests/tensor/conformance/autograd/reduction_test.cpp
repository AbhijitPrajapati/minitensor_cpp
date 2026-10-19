#include <minitensor/autograd.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>

#include <gtest/gtest.h>

#include "tensor/conformance/backend.hpp"
#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test::conformance
{
TEST(ReductionAutogradTest, BroadcastsSumCotangents)
{
    const Tensor input = full(Shape{2, 3}, 0.0F);
    const Tensor output = sum(input, {1}, true);
    const Tensor seed = from_data({2.0F, 4.0F}, Shape{2, 1});
    const Tensor result = vjp(output, input, seed).front();
    const std::array<float, 6> expected{2.0F, 2.0F, 2.0F, 4.0F, 4.0F, 4.0F};
    expect_tensor_eq(result, expected);
}

TEST(ReductionAutogradTest, DifferentiatesLogSumExp)
{
    const Tensor input = from_data({0.0F, 1.0F, 2.0F, 0.0F, -1.0F, -2.0F}, Shape{2, 3});
    const Tensor result = grad(sum(logsumexp(input, {1})), input).front();

    const std::array<float, 6> expected{0.0900306F, 0.2447285F, 0.6652410F, 0.6652410F, 0.2447285F, 0.0900306F};
    expect_tensor_near(result, expected);
}
} // namespace minitensor::test::conformance
