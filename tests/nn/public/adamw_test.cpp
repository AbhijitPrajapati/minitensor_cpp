#include <minitensor/nn/optimizers/adamw.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/leafification.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>

namespace minitensor::nn::test
{
namespace
{
class ParameterPair final
{
  public:
    ParameterPair(Parameter first, Parameter second) : first_{std::move(first)}, second_{std::move(second)}
    {
    }

    [[nodiscard]] const Parameter &first() const noexcept
    {
        return first_;
    }

    [[nodiscard]] const Parameter &second() const noexcept
    {
        return second_;
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter("first", self.first_);
        visitor.parameter("second", self.second_);
    }

    Parameter first_;
    Parameter second_;

    friend class ::minitensor::nn::ParameterTreeAccess;
};

class MixedParameterTree final
{
  public:
    MixedParameterTree(Parameter shared, Parameter frozen)
        : first_{shared}, second_{std::move(shared)}, frozen_{std::move(frozen)}
    {
    }

    [[nodiscard]] const Parameter &first() const noexcept
    {
        return first_;
    }

    [[nodiscard]] const Parameter &second() const noexcept
    {
        return second_;
    }

    [[nodiscard]] const Parameter &frozen() const noexcept
    {
        return frozen_;
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter("first", self.first_);
        visitor.parameter("second", self.second_);
        visitor.parameter("frozen", self.frozen_);
    }

    Parameter first_;
    Parameter second_;
    Parameter frozen_;

    friend class ::minitensor::nn::ParameterTreeAccess;
};
} // namespace

TEST(AdamWTest, ValidatesHyperparameters)
{
    const AdamW optimizer{0.001F, 0.01F};
    EXPECT_FLOAT_EQ(optimizer.learning_rate(), 0.001F);
    EXPECT_FLOAT_EQ(optimizer.weight_decay(), 0.01F);
    EXPECT_FLOAT_EQ(optimizer.beta1(), 0.9F);
    EXPECT_FLOAT_EQ(optimizer.beta2(), 0.999F);
    EXPECT_FLOAT_EQ(optimizer.epsilon(), 1e-8F);

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW((void)(AdamW{-0.1F, 0.0F}), std::invalid_argument);
    EXPECT_THROW((void)(AdamW{0.1F, -0.1F}), std::invalid_argument);
    EXPECT_THROW((void)(AdamW{0.1F, 0.0F, 1.0F}), std::invalid_argument);
    EXPECT_THROW((void)(AdamW{0.1F, 0.0F, 0.9F, nan}), std::invalid_argument);
    EXPECT_THROW((void)(AdamW{0.1F, 0.0F, 0.9F, 0.999F, 0.0F}), std::invalid_argument);
    EXPECT_THROW((void)(AdamW{infinity, 0.0F}), std::invalid_argument);
}

TEST(AdamWTest, AppliesOneGlobalBiasCorrectedStep)
{
    const Parameter first{from_data({1.0F}, Shape{})};
    const Parameter second{from_data({-2.0F}, Shape{})};
    const ParameterPair parameters{first, second};
    ParameterGradients gradients;
    gradients.emplace(first.id(), from_data({2.0F}, Shape{}));
    gradients.emplace(second.id(), from_data({-4.0F}, Shape{}));
    const AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};

    auto [updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), gradients);

    const float decay = 1.0F - 0.1F * 0.2F;
    const float expected_first = decay * 1.0F - 0.1F * 2.0F / 2.01F;
    const float expected_second = decay * -2.0F + 0.1F * 4.0F / 4.01F;
    EXPECT_EQ(state.step(), 1U);
    EXPECT_NEAR(to_vector(updated.first().value()).front(), expected_first, 1e-6F);
    EXPECT_NEAR(to_vector(updated.second().value()).front(), expected_second, 1e-6F);
    EXPECT_FLOAT_EQ(to_vector(parameters.first().value()).front(), 1.0F);
    EXPECT_FLOAT_EQ(to_vector(parameters.second().value()).front(), -2.0F);
}

TEST(AdamWTest, AccumulatesAndLeafifiesMomentState)
{
    const Parameter first{from_data({1.0F}, Shape{})};
    const Parameter frozen{from_data({8.0F}, Shape{}), ParameterMetadata{.trainable = false}};
    const ParameterPair parameters{first, frozen};
    const AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};
    ParameterGradients first_gradients;
    first_gradients.emplace(first.id(), from_data({2.0F}, Shape{}));

    auto [once_updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), first_gradients);
    state = leafify_optimizer_state(std::move(state));
    EXPECT_EQ(state.step(), 1U);

    ParameterGradients second_gradients;
    second_gradients.emplace(first.id(), from_data({1.0F}, Shape{}));
    auto [twice_updated, next_state] =
        optimizer.step(leafify_parameter_values(std::move(once_updated)), std::move(state), second_gradients);

    const float first_value = 0.98F - 0.1F * 2.0F / 2.01F;
    const float corrected_first_moment = 1.0F / 0.75F;
    const float corrected_second_moment = 1.5F / 0.9375F;
    const float expected =
        0.98F * first_value - 0.1F * corrected_first_moment / (std::sqrt(corrected_second_moment) + 0.01F);

    EXPECT_EQ(next_state.step(), 2U);
    EXPECT_NEAR(to_vector(twice_updated.first().value()).front(), expected, 1e-6F);
    EXPECT_FLOAT_EQ(to_vector(twice_updated.second().value()).front(), 8.0F);
}

TEST(AdamWTest, HandlesTiesAndRejectsMismatchedInputs)
{
    const Parameter shared{from_data({1.0F, 2.0F}, Shape{2})};
    const Parameter frozen{from_data({3.0F, 4.0F}, Shape{2}), ParameterMetadata{.trainable = false}};
    const MixedParameterTree parameters{shared, frozen};
    ParameterGradients gradients;
    gradients.emplace(shared.id(), from_data({0.5F, 1.0F}, Shape{2}));
    const AdamW optimizer{0.1F, 0.0F};

    auto [updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), gradients);
    EXPECT_EQ(updated.first().id(), updated.second().id());
    EXPECT_EQ(to_vector(updated.first().value()), to_vector(updated.second().value()));
    EXPECT_EQ(to_vector(updated.frozen().value()), (std::vector<float>{3.0F, 4.0F}));

    ParameterGradients mismatched = gradients;
    mismatched.insert_or_assign(shared.id(), from_data({1.0F}, Shape{}));
    EXPECT_THROW((void)optimizer.step(parameters, state, mismatched), std::invalid_argument);

    const Parameter other{from_data({1.0F, 2.0F}, Shape{2})};
    const MixedParameterTree other_parameters{other, frozen};
    ParameterGradients other_gradients;
    other_gradients.emplace(other.id(), from_data({0.5F, 1.0F}, Shape{2}));
    EXPECT_THROW((void)optimizer.step(other_parameters, state, other_gradients), std::invalid_argument);
}
} // namespace minitensor::nn::test
