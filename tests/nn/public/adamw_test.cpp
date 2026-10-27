#include <minitensor/nn/optimizers/adamw.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/leafification.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/creation.hpp>

#include "nn/support/parameter_trees.hpp"

namespace minitensor::nn::test
{
using support::MixedParameterTree;
using support::ParameterPair;

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
    ParameterBuilder builder;
    const ParameterId first = builder.add(ParameterState{from_data({1.0F}, Shape{})});
    const ParameterId second = builder.add(ParameterState{from_data({-2.0F}, Shape{})});
    const auto parameters = std::move(builder).finish(ParameterPair{first, second});
    ParameterGradients gradients;
    gradients.emplace(first, from_data({2.0F}, Shape{}));
    gradients.emplace(second, from_data({-4.0F}, Shape{}));
    const AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};

    auto [updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), gradients);

    const float decay = 1.0F - 0.1F * 0.2F;
    const float expected_first = decay * 1.0F - 0.1F * 2.0F / 2.01F;
    const float expected_second = decay * -2.0F + 0.1F * 4.0F / 4.01F;
    EXPECT_EQ(state.step(), 1U);
    EXPECT_NEAR(to_vector(updated.value(updated.tree().first())).front(), expected_first, 1e-6F);
    EXPECT_NEAR(to_vector(updated.value(updated.tree().second())).front(), expected_second, 1e-6F);
    EXPECT_FLOAT_EQ(to_vector(parameters.value(parameters.tree().first())).front(), 1.0F);
    EXPECT_FLOAT_EQ(to_vector(parameters.value(parameters.tree().second())).front(), -2.0F);
}

TEST(AdamWTest, AccumulatesAndLeafifiesMomentState)
{
    ParameterBuilder builder;
    const ParameterId first = builder.add(ParameterState{from_data({1.0F}, Shape{})});
    const ParameterId frozen =
        builder.add(ParameterState{from_data({8.0F}, Shape{}), ParameterMetadata{.trainable = false}});
    const auto parameters = std::move(builder).finish(ParameterPair{first, frozen});
    const AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};
    ParameterGradients first_gradients;
    first_gradients.emplace(first, from_data({2.0F}, Shape{}));

    auto [once_updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), first_gradients);
    state = leafify_optimizer_state(std::move(state));
    EXPECT_EQ(state.step(), 1U);

    ParameterGradients second_gradients;
    second_gradients.emplace(first, from_data({1.0F}, Shape{}));
    auto [twice_updated, next_state] =
        optimizer.step(leafify_parameter_values(std::move(once_updated)), std::move(state), second_gradients);

    const float first_value = 0.98F - 0.1F * 2.0F / 2.01F;
    const float corrected_first_moment = 1.0F / 0.75F;
    const float corrected_second_moment = 1.5F / 0.9375F;
    const float expected =
        0.98F * first_value - 0.1F * corrected_first_moment / (std::sqrt(corrected_second_moment) + 0.01F);

    EXPECT_EQ(next_state.step(), 2U);
    EXPECT_NEAR(to_vector(twice_updated.value(twice_updated.tree().first())).front(), expected, 1e-6F);
    EXPECT_FLOAT_EQ(to_vector(twice_updated.value(twice_updated.tree().second())).front(), 8.0F);
}

TEST(AdamWTest, HandlesTiesAndRejectsMismatchedInputs)
{
    ParameterBuilder builder;
    const ParameterId shared = builder.add(ParameterState{from_data({1.0F, 2.0F}, Shape{2})});
    const ParameterId frozen =
        builder.add(ParameterState{from_data({3.0F, 4.0F}, Shape{2}), ParameterMetadata{.trainable = false}});
    const auto parameters = std::move(builder).finish(MixedParameterTree{shared, frozen});
    ParameterGradients gradients;
    gradients.emplace(shared, from_data({0.5F, 1.0F}, Shape{2}));
    const AdamW optimizer{0.1F, 0.0F};

    auto [updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), gradients);
    EXPECT_EQ(updated.tree().first(), updated.tree().second());
    EXPECT_EQ(to_vector(updated.value(updated.tree().first())), to_vector(updated.value(updated.tree().second())));
    EXPECT_EQ(to_vector(updated.value(updated.tree().frozen())), (std::vector<float>{3.0F, 4.0F}));

    ParameterGradients mismatched = gradients;
    mismatched.insert_or_assign(shared, from_data({1.0F}, Shape{}));
    EXPECT_THROW((void)optimizer.step(parameters, state, mismatched), std::invalid_argument);

    ParameterBuilder other_builder;
    const auto other = other_builder.add({from_data({1.0F, 2.0F}, Shape{2})});
    const auto other_frozen = other_builder.add({from_data({3.0F, 4.0F}, Shape{2}), {.trainable = false}});
    const auto other_parameters = std::move(other_builder).finish(MixedParameterTree{other, other_frozen});
    ParameterGradients other_gradients;
    other_gradients.emplace(other, from_data({0.5F, 1.0F}, Shape{2}));
    EXPECT_THROW((void)optimizer.step(other_parameters, state, other_gradients), std::invalid_argument);

    const auto reshaped = parameters.transform([&](ParameterId id, const ParameterState &parameter) {
        return ParameterState{id == shared ? full(Shape{1, 2}, 1.0F) : parameter.value, parameter.metadata};
    });
    const ParameterGradients reshaped_gradients{{shared, full(Shape{1, 2}, 1.0F)}};
    EXPECT_THROW((void)optimizer.step(reshaped, state, reshaped_gradients), std::invalid_argument);

    const auto all_frozen = std::move(parameters).transform([](ParameterId, const ParameterState &parameter) {
        return ParameterState{parameter.value, {.trainable = false}};
    });
    EXPECT_THROW((void)optimizer.step(all_frozen, state, {}), std::invalid_argument);
    EXPECT_EQ(state.step(), 1U);
}
} // namespace minitensor::nn::test
