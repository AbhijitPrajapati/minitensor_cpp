#include <minitensor/persistence/training_state.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/persistence/optimizer_configuration.hpp>

#include "nn/support/parameter_trees.hpp"
#include "persistence/support/record.hpp"

namespace minitensor::persistence::test
{
using nn::test::support::MixedParameterTree;
using support::replace_field;

namespace
{
nn::ParameterSnapshot<MixedParameterTree> make_snapshot(bool trainable = true)
{
    nn::ParameterBuilder builder;
    const auto shared = builder.add({from_data({1.0F}, Shape{}), {.trainable = trainable}});
    const auto frozen = builder.add({from_data({8.0F}, Shape{}), {.trainable = !trainable}});
    return std::move(builder).finish(MixedParameterTree{shared, frozen});
}
} // namespace

TEST(TrainingStatePersistenceTest, ResumesAdamWWithTiedParametersAndFreshIds)
{
    const auto source = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};
    const nn::ParameterGradients first_gradients{{source.tree().first(), from_data({2.0F}, Shape{})}};
    const auto [updated, state] = optimizer.step(source, optimizer.initialize(source), first_gradients);

    Record checkpoint;
    checkpoint.add("training", capture_training_state<nn::AdamW>(updated, state));
    checkpoint.add("optimizer_config", capture_optimizer_config(optimizer));
    const auto &training = checkpoint.get("training").get<Record>();
    EXPECT_TRUE(training.contains({"kind", "version", "parameters", "optimizer_state"}, true));

    const auto target = make_snapshot(false);
    const auto target_id = target.tree().first();
    ASSERT_NE(target_id, source.tree().first());
    const auto restored = restore_training_state<nn::AdamW>(training, target);
    const auto restored_optimizer =
        restore_optimizer_config<nn::AdamW>(checkpoint.get("optimizer_config").get<Record>());

    EXPECT_EQ(restored.snapshot.tree().first(), target_id);
    EXPECT_EQ(restored.snapshot.tree().second(), target_id);
    EXPECT_TRUE(restored.snapshot.get(target_id).metadata.trainable);
    EXPECT_FALSE(restored.snapshot.get(target.tree().frozen()).metadata.trainable);
    EXPECT_EQ(restored.optimizer_state.step(), 1U);
    EXPECT_FLOAT_EQ(item(restored.snapshot.value(target_id)), item(updated.value(source.tree().first())));

    const auto next_gradient = from_data({1.0F}, Shape{});
    const nn::ParameterGradients expected_gradients{{source.tree().first(), next_gradient}};
    const nn::ParameterGradients restored_gradients{{target_id, next_gradient}};
    const auto [expected, expected_state] = optimizer.step(updated, state, expected_gradients);
    const auto [actual, actual_state] =
        restored_optimizer.step(restored.snapshot, restored.optimizer_state, restored_gradients);

    EXPECT_EQ(actual_state.step(), expected_state.step());
    EXPECT_FLOAT_EQ(item(actual.value(target_id)), item(expected.value(source.tree().first())));
    EXPECT_FLOAT_EQ(item(actual.value(target.tree().frozen())), 8.0F);
}

TEST(TrainingStatePersistenceTest, RejectsOptimizerStateForDifferentParameters)
{
    const auto source = make_snapshot();
    const auto other = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto state = optimizer.initialize(source);

    EXPECT_THROW((void)capture_training_state<nn::AdamW>(other, state), std::invalid_argument);
}

TEST(TrainingStatePersistenceTest, RejectsInvalidTrainingRecords)
{
    const auto snapshot = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto record = capture_training_state<nn::AdamW>(snapshot, optimizer.initialize(snapshot));

    EXPECT_THROW((void)restore_training_state<nn::AdamW>(replace_field(record, "kind", "parameter_snapshot"), snapshot),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_training_state<nn::AdamW>(replace_field(record, "version", std::uint64_t{1}), snapshot),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_training_state<nn::AdamW>(replace_field(record, "optimizer_state", true), snapshot),
                 std::logic_error);
}

TEST(TrainingStatePersistenceTest, RejectsInvalidMomentReferences)
{
    const auto snapshot = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto record = capture_training_state<nn::AdamW>(snapshot, optimizer.initialize(snapshot));
    const auto &state = record.get("optimizer_state").get<Record>();
    const auto &moments = state.get("moments").get<Value::List>();
    ASSERT_EQ(moments.size(), 1U);
    const auto &moment = moments.front().get<Record>();

    const auto with_moments = [&](const Value::List &entries) {
        return replace_field(record, "optimizer_state", replace_field(state, "moments", entries));
    };

    EXPECT_THROW((void)restore_training_state<nn::AdamW>(with_moments({}), snapshot), std::invalid_argument);
    EXPECT_THROW((void)restore_training_state<nn::AdamW>(with_moments({moment, moment}), snapshot),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_training_state<nn::AdamW>(
                     with_moments({replace_field(moment, "parameter", std::uint64_t{1})}), snapshot),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_training_state<nn::AdamW>(
                     with_moments({replace_field(moment, "parameter", std::uint64_t{2})}), snapshot),
                 std::out_of_range);
    EXPECT_THROW((void)restore_training_state<nn::AdamW>(
                     with_moments({replace_field(moment, "first", from_data({0.0F}, Shape{1}))}), snapshot),
                 std::invalid_argument);
}
} // namespace minitensor::persistence::test
