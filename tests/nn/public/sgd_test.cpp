#include <minitensor/nn/optimizers/sgd.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/nn/module.hpp>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/nn/persistence/checkpoint.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/persistence/record.hpp>
#include <minitensor/random.hpp>

#include "nn/support/parameter_trees.hpp"
#include "persistence/support/record.hpp"

namespace minitensor::nn::test
{
using support::MixedParameterTree;

TEST(SGDTest, ValidatesLearningRate)
{
    const SGD zero{0.0F};
    EXPECT_FLOAT_EQ(zero.learning_rate(), 0.0F);

    EXPECT_THROW((void)SGD{-0.1F}, std::invalid_argument);
    EXPECT_THROW((void)SGD{std::numeric_limits<float>::infinity()}, std::invalid_argument);
    EXPECT_THROW((void)SGD{std::numeric_limits<float>::quiet_NaN()}, std::invalid_argument);
}

TEST(SGDTest, AppliesGradientsWithoutChangingTheSourceTree)
{
    const Linear linear{2, 1};
    const auto parameters = initialize(linear, RandomKey{61});
    ASSERT_NE(parameters.tree().bias(), nullptr);
    const std::vector<float> source_weight = to_vector(parameters.value(parameters.tree().weight()));
    const std::vector<float> source_bias = to_vector(parameters.value(*parameters.tree().bias()));
    const Tensor loss =
        sum(parameters.value(parameters.tree().weight()) * parameters.value(parameters.tree().weight())) +
        sum(parameters.value(*parameters.tree().bias()) * parameters.value(*parameters.tree().bias()));
    const ParameterGradients gradients = parameter_gradients(loss, parameters);
    const SGD optimizer{0.25F};

    auto [updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), gradients);
    state = leafify_optimizer_state(std::move(state));

    EXPECT_EQ(updated.tree().weight(), parameters.tree().weight());
    ASSERT_NE(updated.tree().bias(), nullptr);
    EXPECT_EQ(*updated.tree().bias(), *parameters.tree().bias());
    const std::vector<float> updated_weight = to_vector(updated.value(updated.tree().weight()));
    const std::vector<float> updated_bias = to_vector(updated.value(*updated.tree().bias()));
    ASSERT_EQ(updated_weight.size(), source_weight.size());
    ASSERT_EQ(updated_bias.size(), source_bias.size());
    for (std::size_t index = 0; index < source_weight.size(); ++index)
    {
        EXPECT_FLOAT_EQ(updated_weight[index], 0.5F * source_weight[index]);
    }
    for (std::size_t index = 0; index < source_bias.size(); ++index)
    {
        EXPECT_FLOAT_EQ(updated_bias[index], 0.5F * source_bias[index]);
    }
    EXPECT_EQ(to_vector(parameters.value(parameters.tree().weight())), source_weight);
    EXPECT_EQ(to_vector(parameters.value(*parameters.tree().bias())), source_bias);
}

TEST(SGDTest, UpdatesTiedParametersOnceAndLeavesFrozenParametersUnchanged)
{
    ParameterBuilder builder;
    const ParameterId shared = builder.add(ParameterState{from_data({1.0F, 2.0F}, Shape{2})});
    const ParameterId frozen =
        builder.add(ParameterState{from_data({3.0F, 4.0F}, Shape{2}), ParameterMetadata{.trainable = false}});
    const auto parameters = std::move(builder).finish(MixedParameterTree{shared, frozen});
    ParameterGradients gradients;
    gradients.emplace(shared, from_data({0.5F, 1.0F}, Shape{2}));
    const SGD optimizer{0.5F};

    auto [updated, state] = optimizer.step(parameters, optimizer.initialize(parameters), gradients);
    (void)state;

    EXPECT_EQ(updated.tree().first(), shared);
    EXPECT_EQ(updated.tree().second(), shared);
    EXPECT_EQ(to_vector(updated.value(updated.tree().first())), (std::vector<float>{0.75F, 1.5F}));
    EXPECT_EQ(to_vector(updated.value(updated.tree().second())), (std::vector<float>{0.75F, 1.5F}));
    EXPECT_EQ(updated.tree().frozen(), frozen);
    EXPECT_FALSE(updated.get(updated.tree().frozen()).metadata.trainable);
    EXPECT_EQ(to_vector(updated.value(updated.tree().frozen())), (std::vector<float>{3.0F, 4.0F}));

    gradients.emplace(frozen, parameters.value(frozen));
    EXPECT_THROW((void)optimizer.step(parameters, state, gradients), std::invalid_argument);
}

TEST(SGDTest, RejectsIncompleteMismatchedAndExtraGradientMaps)
{
    ParameterBuilder builder;
    const Linear linear{2, 1};
    const auto parameters = initialize(linear, RandomKey{67});
    ASSERT_NE(parameters.tree().bias(), nullptr);
    const Tensor loss =
        sum(parameters.value(parameters.tree().weight()) * parameters.value(parameters.tree().weight())) +
        sum(parameters.value(*parameters.tree().bias()) * parameters.value(*parameters.tree().bias()));
    const ParameterGradients valid = parameter_gradients(loss, parameters);
    const SGD optimizer{0.1F};
    const SGD::State state = optimizer.initialize(parameters);

    ParameterGradients missing = valid;
    missing.erase(*parameters.tree().bias());
    EXPECT_THROW((void)optimizer.step(parameters, state, missing), std::invalid_argument);

    ParameterGradients mismatched = valid;
    mismatched.insert_or_assign(parameters.tree().weight(), from_data({1.0F}, Shape{}));
    EXPECT_THROW((void)optimizer.step(parameters, state, mismatched), std::invalid_argument);

    ParameterGradients extra = valid;
    const ParameterId unrelated = builder.add(ParameterState{from_data({1.0F}, Shape{})});
    extra.emplace(unrelated, from_data({1.0F}, Shape{}));
    EXPECT_THROW((void)optimizer.step(parameters, state, extra), std::invalid_argument);
}

TEST(SGDTest, RejectsGradientDTypeAndDeviceMismatches)
{
    const Linear linear{2, 1, false};
    const auto parameters = initialize(linear, RandomKey{71});
    const auto id = parameters.tree().weight();
    const SGD optimizer{0.1F};
    const auto state = optimizer.initialize(parameters);

    ParameterGradients gradients{{id, full(Shape{1, 2}, 1, TensorOptions{DType::Int64, Device::cpu()})}};
    EXPECT_THROW((void)optimizer.step(parameters, state, gradients), std::invalid_argument);
    gradients.insert_or_assign(id, full(Shape{1, 2}, 1.0F, TensorOptions{DType::Float32, Device::cpu(1)}));
    EXPECT_THROW((void)optimizer.step(parameters, state, gradients), std::invalid_argument);
}

TEST(SGDTest, RejectsIntegralTrainableParametersAndPreservesFrozenIntegralParameters)
{
    ParameterBuilder builder;
    const auto shared = builder.add({full(Shape{2}, 1, TensorOptions{DType::Int64})});
    const auto frozen = builder.add({full(Shape{2}, 3, TensorOptions{DType::Int64}), {.trainable = false}});
    const auto parameters = std::move(builder).finish(MixedParameterTree{shared, frozen});
    const SGD optimizer{0.1F};
    EXPECT_THROW((void)optimizer.step(parameters, optimizer.initialize(parameters), {}), std::invalid_argument);

    const auto all_frozen = std::move(parameters).transform([](ParameterId, const ParameterState &state) {
        return ParameterState{state.value, {.trainable = false}};
    });
    const auto [updated, state] = optimizer.step(all_frozen, optimizer.initialize(all_frozen), {});
    (void)state;
    EXPECT_EQ(to_vector<std::int64_t>(updated.value(shared)), (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(to_vector<std::int64_t>(updated.value(frozen)), (std::vector<std::int64_t>{3, 3}));
}

TEST(SGDCheckpointPersistenceTest, ResumesWithRestoredParametersAndLearningRate)
{
    const Linear linear{1, 1, false};
    const auto source = initialize(linear, RandomKey{71});
    const auto target = initialize(linear, RandomKey{72});
    const auto source_id = source.tree().weight();
    const auto target_id = target.tree().weight();
    ASSERT_NE(source_id, target_id);
    const SGD optimizer{0.125F};
    const auto record = capture_checkpoint(source, optimizer, optimizer.initialize(source));
    const auto [restored, restored_optimizer, restored_state] = restore_sgd_checkpoint(record, target);

    EXPECT_EQ(restored.tree().weight(), target_id);
    EXPECT_FLOAT_EQ(item(restored.value(target_id)), item(source.value(source_id)));
    EXPECT_FLOAT_EQ(restored_optimizer.learning_rate(), optimizer.learning_rate());

    const auto gradient = from_data({2.0F}, Shape{1, 1});
    const auto [updated, state] = restored_optimizer.step(restored, restored_state, {{target_id, gradient}});
    EXPECT_FLOAT_EQ(item(updated.value(target_id)), item(source.value(source_id)) - 0.25F);
}

TEST(SGDCheckpointPersistenceTest, RejectsInvalidLearningRate)
{
    using persistence::Record;
    using persistence::test::support::replace_field;

    const auto snapshot = ParameterBuilder{}.finish(support::EmptyTree{});
    const SGD optimizer{0.125F};
    const auto record = capture_checkpoint(snapshot, optimizer, optimizer.initialize(snapshot));
    const auto &sgd = record.get("sgd").get<Record>();
    const auto invalid = replace_field(sgd, "learning_rate", -1.0F);

    EXPECT_THROW((void)restore_sgd_checkpoint(replace_field(record, "sgd", invalid), snapshot), std::invalid_argument);
}
} // namespace minitensor::nn::test
