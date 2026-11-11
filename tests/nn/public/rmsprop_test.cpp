#include <minitensor/nn/optimizers/rmsprop.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/nn/leafification.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>

#include "nn/support/parameter_trees.hpp"

namespace minitensor::nn::test
{
using support::MixedParameterTree;
using support::ParameterPair;

TEST(RMSPropTest, ValidatesHyperparameters)
{
    const RMSProp optimizer{0.01F};
    EXPECT_FLOAT_EQ(optimizer.learning_rate(), 0.01F);
    EXPECT_FLOAT_EQ(optimizer.alpha(), 0.9F);
    EXPECT_FLOAT_EQ(optimizer.epsilon(), 1e-8F);
    EXPECT_NO_THROW((void)(RMSProp{0.0F, 0.0F, 0.1F}));

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float invalid : {-0.1F, infinity, -infinity, nan})
    {
        EXPECT_THROW((void)(RMSProp{invalid}), std::invalid_argument);
    }
    for (float invalid : {-0.1F, 1.0F, infinity, -infinity, nan})
    {
        EXPECT_THROW((void)(RMSProp{0.1F, invalid}), std::invalid_argument);
    }
    for (float invalid : {-0.1F, 0.0F, infinity, -infinity, nan})
    {
        EXPECT_THROW((void)(RMSProp{0.1F, 0.9F, invalid}), std::invalid_argument);
    }
}

TEST(RMSPropTest, AppliesSquaredGradientAverageAndEpsilonOutsideSquareRoot)
{
    ParameterBuilder builder;
    const auto first = builder.add({from_data({1.0F, -2.0F}, Shape{2})});
    const auto second = builder.add({from_data({3.0F}, Shape{})});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, second});
    const RMSProp optimizer{0.1F, 0.75F, 0.25F};
    const ParameterGradients gradients{{first, from_data({2.0F, -4.0F}, Shape{2})},
                                       {second, from_data({0.0F}, Shape{})}};

    const auto [updated, state] = optimizer.step(snapshot, optimizer.initialize(snapshot), gradients);
    const auto values = to_vector(updated.value(first));
    ASSERT_EQ(values.size(), 2U);
    EXPECT_NEAR(values[0], 1.0F - 0.1F * 2.0F / (std::sqrt(1.0F) + 0.25F), 1e-6F);
    EXPECT_NEAR(values[1], -2.0F + 0.1F * 4.0F / (std::sqrt(4.0F) + 0.25F), 1e-6F);
    EXPECT_FLOAT_EQ(item(updated.value(second)), 3.0F);
    EXPECT_EQ(updated.tree().first(), first);
    EXPECT_EQ(updated.tree().second(), second);
    EXPECT_EQ(to_vector(snapshot.value(first)), (std::vector<float>{1.0F, -2.0F}));
}

TEST(RMSPropTest, AccumulatesHistoryAndKeepsCopiedStatesIndependent)
{
    ParameterBuilder builder;
    const auto first = builder.add({from_data({1.0F}, Shape{})});
    const auto second = builder.add({from_data({-2.0F}, Shape{})});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, second});
    const RMSProp optimizer{0.1F, 0.75F, 0.25F};
    const auto initial = optimizer.initialize(snapshot);
    const ParameterGradients first_gradients{{first, from_data({2.0F}, Shape{})}, {second, from_data({4.0F}, Shape{})}};
    const auto [once_updated, state] = optimizer.step(snapshot, initial, first_gradients);
    const ParameterGradients next_gradients{{first, from_data({1.0F}, Shape{})}, {second, from_data({-2.0F}, Shape{})}};

    const auto [twice_updated, next_state] = optimizer.step(once_updated, state, next_gradients);
    const auto [repeated, repeated_state] = optimizer.step(once_updated, state, next_gradients);
    const float expected_first = 1.0F - 0.2F / 1.25F - 0.1F / (std::sqrt(1.0F) + 0.25F);
    const float expected_second = -2.0F - 0.4F / 2.25F + 0.2F / (std::sqrt(4.0F) + 0.25F);
    EXPECT_NEAR(item(twice_updated.value(first)), expected_first, 1e-6F);
    EXPECT_NEAR(item(twice_updated.value(second)), expected_second, 1e-6F);
    EXPECT_FLOAT_EQ(item(repeated.value(first)), item(twice_updated.value(first)));
    EXPECT_FLOAT_EQ(item(repeated.value(second)), item(twice_updated.value(second)));

    const auto [restarted, restarted_state] = optimizer.step(once_updated, initial, next_gradients);
    EXPECT_NEAR(item(restarted.value(first)), item(once_updated.value(first)) - 0.1F / 0.75F, 1e-6F);
}

TEST(RMSPropTest, UpdatesTiedParametersOnceAndPreservesFrozenParameters)
{
    ParameterBuilder builder;
    const auto shared = builder.add({from_data({1.0F}, Shape{})});
    const auto frozen = builder.add({from_data({8.0F}, Shape{}), {.trainable = false}});
    const auto snapshot = std::move(builder).finish(MixedParameterTree{shared, frozen});
    const RMSProp optimizer{0.1F, 0.75F, 0.25F};
    const auto state = optimizer.initialize(snapshot);
    const ParameterGradients gradients{{shared, from_data({2.0F}, Shape{})}};
    const auto [updated, next_state] = optimizer.step(snapshot, state, gradients);

    EXPECT_EQ(updated.tree().first(), updated.tree().second());
    EXPECT_NEAR(item(updated.value(shared)), 1.0F - 0.2F / 1.25F, 1e-6F);
    EXPECT_FLOAT_EQ(item(updated.value(frozen)), 8.0F);
    EXPECT_FALSE(updated.get(frozen).metadata.trainable);

    auto extra = gradients;
    extra.emplace(frozen, from_data({1.0F}, Shape{}));
    EXPECT_THROW((void)optimizer.step(snapshot, state, extra), std::invalid_argument);
}

TEST(RMSPropTest, RejectsMissingExtraAndMismatchedGradients)
{
    ParameterBuilder builder;
    const auto first = builder.add({from_data({1.0F, 2.0F}, Shape{2})});
    const auto second = builder.add({from_data({3.0F, 4.0F}, Shape{2})});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, second});
    const RMSProp optimizer{0.1F};
    const auto state = optimizer.initialize(snapshot);
    const ParameterGradients gradients{{first, full(Shape{2}, 1.0F)}, {second, full(Shape{2}, 1.0F)}};

    auto missing = gradients;
    missing.erase(second);
    EXPECT_THROW((void)optimizer.step(snapshot, state, missing), std::invalid_argument);

    ParameterBuilder extra_builder;
    const auto extra_id = extra_builder.add({full(Shape{2}, 1.0F)});
    auto extra = gradients;
    extra.emplace(extra_id, full(Shape{2}, 1.0F));
    EXPECT_THROW((void)optimizer.step(snapshot, state, extra), std::invalid_argument);

    for (const Tensor &invalid : {full(Shape{}, 1.0F), full(Shape{2}, 1, TensorOptions{DType::Int64}),
                                  full(Shape{2}, 1.0F, TensorOptions{DType::Float32, Device::cpu(1)})})
    {
        auto mismatched = gradients;
        mismatched.insert_or_assign(first, invalid);
        EXPECT_THROW((void)optimizer.step(snapshot, state, mismatched), std::invalid_argument);
    }
    const auto [updated, next_state] = optimizer.step(snapshot, state, gradients);
    EXPECT_LT(to_vector(updated.value(first)).front(), 1.0F);
}

TEST(RMSPropTest, RejectsStateForDifferentIdsOrParameterSpecifications)
{
    ParameterBuilder builder;
    const auto first = builder.add({full(Shape{2}, 1.0F)});
    const auto second = builder.add({full(Shape{2}, 2.0F)});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, second});
    const RMSProp optimizer{0.1F};
    const auto state = optimizer.initialize(snapshot);

    ParameterBuilder other_builder;
    const auto other_first = other_builder.add({full(Shape{2}, 1.0F)});
    const auto other_second = other_builder.add({full(Shape{2}, 2.0F)});
    const auto other = std::move(other_builder).finish(ParameterPair{other_first, other_second});
    const ParameterGradients other_gradients{{other_first, full(Shape{2}, 1.0F)}, {other_second, full(Shape{2}, 1.0F)}};
    EXPECT_THROW((void)optimizer.step(other, state, other_gradients), std::invalid_argument);

    for (const Tensor &value :
         {full(Shape{1, 2}, 1.0F), full(Shape{2}, 1.0F, TensorOptions{DType::Float32, Device::cpu(1)})})
    {
        const auto changed = snapshot.transform([&](ParameterId id, const ParameterState &parameter) {
            return ParameterState{id == first ? value : parameter.value, parameter.metadata};
        });
        const ParameterGradients gradients{
            {first, full(value.shape(), 1.0F, TensorOptions{value.dtype(), value.device()})},
            {second, full(Shape{2}, 1.0F)}};
        EXPECT_THROW((void)optimizer.step(changed, state, gradients), std::invalid_argument);
    }
}

TEST(RMSPropTest, RejectsIntegralTrainableParametersAndSupportsFrozenIntegralParameters)
{
    ParameterBuilder builder;
    const auto first = builder.add({full(Shape{2}, 1, TensorOptions{DType::Int64})});
    const auto second = builder.add({full(Shape{2}, 3, TensorOptions{DType::Int64}), {.trainable = false}});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, second});
    const RMSProp optimizer{0.1F};
    EXPECT_THROW((void)optimizer.initialize(snapshot), std::invalid_argument);

    const auto frozen = snapshot.transform([](ParameterId, const ParameterState &parameter) {
        return ParameterState{parameter.value, {.trainable = false}};
    });
    auto state = leafify_optimizer_state(optimizer.initialize(frozen));
    const auto [updated, next_state] = optimizer.step(frozen, std::move(state), {});
    EXPECT_EQ(to_vector<std::int64_t>(updated.value(first)), (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(to_vector<std::int64_t>(updated.value(second)), (std::vector<std::int64_t>{3, 3}));
}

TEST(RMSPropTest, HandlesEmptySnapshots)
{
    const auto snapshot = ParameterBuilder{}.finish(support::EmptyTree{});
    const RMSProp optimizer{0.1F};
    auto state = leafify_optimizer_state(optimizer.initialize(snapshot));
    const auto [updated, next_state] = optimizer.step(snapshot, std::move(state), {});
    EXPECT_EQ(updated.parameters().size(), 0U);
}

TEST(RMSPropTest, SupportsZeroLearningRateAndZeroAlpha)
{
    ParameterBuilder builder;
    const auto first = builder.add({from_data({1.0F}, Shape{})});
    const auto second = builder.add({from_data({-2.0F}, Shape{})});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, second});
    const ParameterGradients gradients{{first, from_data({2.0F}, Shape{})}, {second, from_data({-4.0F}, Shape{})}};
    const RMSProp zero_rate{0.0F};
    const auto [unchanged, zero_state] = zero_rate.step(snapshot, zero_rate.initialize(snapshot), gradients);
    EXPECT_FLOAT_EQ(item(unchanged.value(first)), 1.0F);
    EXPECT_FLOAT_EQ(item(unchanged.value(second)), -2.0F);

    const RMSProp zero_alpha{0.1F, 0.0F, 0.25F};
    const auto [updated, state] = zero_alpha.step(snapshot, zero_alpha.initialize(snapshot), gradients);
    const auto [next, next_state] = zero_alpha.step(updated, state, gradients);
    EXPECT_NEAR(item(next.value(first)), 1.0F - 2.0F * 0.2F / 2.25F, 1e-6F);
    EXPECT_NEAR(item(next.value(second)), -2.0F + 2.0F * 0.4F / 4.25F, 1e-6F);
}

TEST(RMSPropTest, LeafifiesMovingAveragesAndPreservesTheirValues)
{
    ParameterBuilder builder;
    const auto first = builder.add({from_data({1.0F}, Shape{})});
    const auto frozen = builder.add({from_data({8.0F}, Shape{}), {.trainable = false}});
    const auto snapshot = std::move(builder).finish(ParameterPair{first, frozen});
    const RMSProp optimizer{0.1F, 0.75F, 0.25F};
    const Tensor initial_gradient = from_data({2.0F}, Shape{});
    const auto [updated, state] = optimizer.step(snapshot, optimizer.initialize(snapshot), {{first, initial_gradient}});
    const auto leaf_snapshot = leafify_parameter_values(updated);
    const auto detached_state = leafify_optimizer_state(state);
    const ParameterGradients gradients{{first, from_data({1.0F}, Shape{})}};

    const auto [attached, attached_state] = optimizer.step(leaf_snapshot, state, gradients);
    const auto [detached, next_state] = optimizer.step(leaf_snapshot, detached_state, gradients);
    EXPECT_FLOAT_EQ(item(attached.value(first)), item(detached.value(first)));
    EXPECT_NEAR(item(detached.value(first)), 1.0F - 0.2F / 1.25F - 0.1F / 1.25F, 1e-6F);
    EXPECT_NE(item(grad(attached.value(first), initial_gradient).front()), 0.0F);
    EXPECT_FLOAT_EQ(item(grad(detached.value(first), initial_gradient).front()), 0.0F);
    EXPECT_FLOAT_EQ(item(detached.value(frozen)), 8.0F);
}
} // namespace minitensor::nn::test
