#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "tensor/conformance/backend.hpp"
#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test::conformance
{
TEST(ElementwiseAutogradTest, AccumulatesAndReducesBroadcastContributions)
{
    const Tensor lhs = from_data({1.0F, 2.0F}, Shape{2, 1});
    const Tensor rhs = from_data({10.0F, 20.0F, 30.0F}, Shape{1, 3});
    const std::array<Tensor, 2> inputs{lhs, rhs};
    const std::vector<Tensor> gradients = grad(sum(lhs * rhs + lhs), inputs);

    ASSERT_EQ(gradients.size(), 2);
    const std::array<float, 2> lhs_expected{63.0F, 63.0F};
    const std::array<float, 3> rhs_expected{3.0F, 3.0F, 3.0F};
    EXPECT_EQ(gradients[0].shape(), lhs.shape());
    EXPECT_EQ(gradients[1].shape(), rhs.shape());
    expect_tensor_eq(gradients[0], lhs_expected);
    expect_tensor_eq(gradients[1], rhs_expected);
}

TEST(ElementwiseAutogradTest, DifferentiatesSubtractionAndDivisionWithBroadcasting)
{
    const Tensor dividend = from_data({2.0F, 4.0F}, Shape{2, 1});
    const Tensor divisor = from_data({1.0F, 2.0F, 4.0F}, Shape{1, 3});
    const std::array<Tensor, 2> inputs{dividend, divisor};
    const std::vector<Tensor> gradients = grad(sum(dividend / divisor - dividend), inputs);

    const std::array<float, 2> dividend_expected{-1.25F, -1.25F};
    const std::array<float, 3> divisor_expected{-6.0F, -1.5F, -0.375F};
    expect_tensor_near(gradients[0], dividend_expected);
    expect_tensor_near(gradients[1], divisor_expected);
}

namespace
{
enum class UnaryRule
{
    Negate,
    Exponential,
    Logarithm,
    SquareRoot,
    HyperbolicTangent,
    AbsoluteValue,
    Sign,
    Sigmoid
};

Tensor apply(UnaryRule rule, const Tensor &input)
{
    switch (rule)
    {
    case UnaryRule::Negate:
        return -input;
    case UnaryRule::Exponential:
        return exp(input);
    case UnaryRule::Logarithm:
        return log(input);
    case UnaryRule::SquareRoot:
        return sqrt(input);
    case UnaryRule::HyperbolicTangent:
        return tanh(input);
    case UnaryRule::AbsoluteValue:
        return abs(input);
    case UnaryRule::Sign:
        return sign(input);
    case UnaryRule::Sigmoid:
        return sigmoid(input);
    }
    throw std::logic_error{"unknown unary VJP test rule"};
}

std::string unary_rule_name(const testing::TestParamInfo<UnaryRule> &info)
{
    switch (info.param)
    {
    case UnaryRule::Negate:
        return "Negate";
    case UnaryRule::Exponential:
        return "Exponential";
    case UnaryRule::Logarithm:
        return "Logarithm";
    case UnaryRule::SquareRoot:
        return "SquareRoot";
    case UnaryRule::HyperbolicTangent:
        return "HyperbolicTangent";
    case UnaryRule::AbsoluteValue:
        return "AbsoluteValue";
    case UnaryRule::Sign:
        return "Sign";
    case UnaryRule::Sigmoid:
        return "Sigmoid";
    }
    return "Unknown";
}
} // namespace

class UnaryAutogradTest : public testing::TestWithParam<UnaryRule>
{
};

TEST_P(UnaryAutogradTest, AppliesTheAnalyticalRule)
{
    const Tensor input = from_data({1.0F, 2.0F}, Shape{2});
    const Tensor gradient = grad(sum(apply(GetParam(), input)), input).front();
    std::array<float, 2> expected{};

    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        const float value = static_cast<float>(index + 1);
        switch (GetParam())
        {
        case UnaryRule::Negate:
            expected[index] = -1.0F;
            break;
        case UnaryRule::Exponential:
            expected[index] = std::exp(value);
            break;
        case UnaryRule::Logarithm:
            expected[index] = 1.0F / value;
            break;
        case UnaryRule::SquareRoot:
            expected[index] = 0.5F / std::sqrt(value);
            break;
        case UnaryRule::HyperbolicTangent: {
            const float output = std::tanh(value);
            expected[index] = 1.0F - output * output;
            break;
        }
        case UnaryRule::AbsoluteValue:
            expected[index] = 1.0F;
            break;
        case UnaryRule::Sign:
            expected[index] = 0.0F;
            break;
        case UnaryRule::Sigmoid: {
            const float output = 1.0F / (1.0F + std::exp(-value));
            expected[index] = output * (1.0F - output);
            break;
        }
        }
    }

    expect_tensor_near(gradient, expected);
}

INSTANTIATE_TEST_SUITE_P(AllRules, UnaryAutogradTest,
                         testing::Values(UnaryRule::Negate, UnaryRule::Exponential, UnaryRule::Logarithm,
                                         UnaryRule::SquareRoot, UnaryRule::HyperbolicTangent, UnaryRule::AbsoluteValue,
                                         UnaryRule::Sign, UnaryRule::Sigmoid),
                         unary_rule_name);

TEST(ElementwiseAutogradTest, LogAddExpReducesBroadcastContributions)
{
    const Tensor lhs = from_data({0.0F, 2.0F}, Shape{2, 1});
    const Tensor rhs = from_data({0.0F, 1.0F}, Shape{1, 2});
    const std::array<Tensor, 2> inputs{lhs, rhs};
    const std::vector<Tensor> gradients = grad(sum(logaddexp(lhs, rhs)), inputs);

    const std::array<float, 2> lhs_expected{0.7689414F, 1.6118556F};
    const std::array<float, 2> rhs_expected{0.6192029F, 1.0F};
    expect_tensor_near(gradients[0], lhs_expected);
    expect_tensor_near(gradients[1], rhs_expected);
}

TEST(ElementwiseAutogradTest, LogAddExpPreservesSmallDominatedGradient)
{
    const Tensor lhs = from_data({20.0F}, Shape{});
    const Tensor rhs = from_data({0.0F}, Shape{});
    const Tensor rhs_gradient = grad(logaddexp(lhs, rhs), rhs).front();

    EXPECT_NEAR(item(rhs_gradient), 2.0611537E-9F, 1.0E-12F);
}

TEST(ElementwiseAutogradTest, AbsoluteValueUsesZeroSubgradientAtOrigin)
{
    const Tensor input = from_data({-2.0F, 0.0F, 3.0F}, Shape{3});
    const Tensor weights = from_data({2.0F, 3.0F, 4.0F}, Shape{3});
    const Tensor gradient = grad(sum(abs(input) * weights), input).front();

    const std::array<float, 3> expected{-2.0F, 0.0F, 4.0F};
    expect_tensor_eq(gradient, expected);
}

TEST(ElementwiseAutogradTest, PreservesScalarLeftOperandOrder)
{
    const Tensor input = from_data({1.0F, 2.0F}, Shape{2});
    const Tensor gradient = grad(sum(4.0F / input), input).front();
    const std::array<float, 2> expected{-4.0F, -1.0F};
    expect_tensor_near(gradient, expected);
}
} // namespace minitensor::test::conformance
