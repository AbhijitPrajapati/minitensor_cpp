#include <minitensor/nn/optimizers/state_persistence/adamw.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_persistence.hpp>
#include <minitensor/persistence/record.hpp>

#include "nn/support/parameter_trees.hpp"
#include "persistence/support/record.hpp"

namespace minitensor::nn::test
{
using persistence::Record;
using persistence::Value;
using persistence::test::support::replace_field;
using support::MixedParameterTree;

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

TEST(AdamWStatePersistenceTest, ResumesAdamWWithTiedParametersAndFreshIds)
{
    const auto source = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};
    const nn::ParameterGradients first_gradients{{source.tree().first(), from_data({2.0F}, Shape{})}};
    const auto [updated, state] = optimizer.step(source, optimizer.initialize(source), first_gradients);

    Record checkpoint;
    checkpoint.add("parameters", capture_snapshot(updated));
    checkpoint.add("optimizer_state", capture_optimizer_state(updated, state));
    checkpoint.add("optimizer_config", capture_optimizer_configuration(optimizer));

    const auto target = make_snapshot(false);
    const auto target_id = target.tree().first();
    ASSERT_NE(target_id, source.tree().first());
    const auto restored = restore_snapshot(checkpoint.get("parameters").get<Record>(), target);
    const auto restored_state = restore_adamw_state(checkpoint.get("optimizer_state").get<Record>(), restored);
    const auto restored_optimizer = restore_adamw_configuration(checkpoint.get("optimizer_config").get<Record>());

    EXPECT_EQ(restored.tree().first(), target_id);
    EXPECT_EQ(restored.tree().second(), target_id);
    EXPECT_TRUE(restored.get(target_id).metadata.trainable);
    EXPECT_FALSE(restored.get(target.tree().frozen()).metadata.trainable);
    EXPECT_EQ(restored_state.step(), 1U);
    EXPECT_FLOAT_EQ(item(restored.value(target_id)), item(updated.value(source.tree().first())));

    const auto next_gradient = from_data({1.0F}, Shape{});
    const nn::ParameterGradients expected_gradients{{source.tree().first(), next_gradient}};
    const nn::ParameterGradients restored_gradients{{target_id, next_gradient}};
    const auto [expected, expected_state] = optimizer.step(updated, state, expected_gradients);
    const auto [actual, actual_state] = restored_optimizer.step(restored, restored_state, restored_gradients);

    EXPECT_EQ(actual_state.step(), expected_state.step());
    EXPECT_FLOAT_EQ(item(actual.value(target_id)), item(expected.value(source.tree().first())));
    EXPECT_FLOAT_EQ(item(actual.value(target.tree().frozen())), 8.0F);
}

TEST(AdamWStatePersistenceTest, RejectsOptimizerStateForDifferentParameters)
{
    const auto source = make_snapshot();
    const auto other = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto state = optimizer.initialize(source);

    EXPECT_THROW((void)capture_optimizer_state(other, state), std::invalid_argument);
}

TEST(AdamWStatePersistenceTest, RejectsInvalidStateRecords)
{
    const auto snapshot = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto record = capture_optimizer_state(snapshot, optimizer.initialize(snapshot));

    EXPECT_THROW((void)restore_adamw_state(replace_field(record, "kind", "snapshot"), snapshot), std::invalid_argument);
    EXPECT_THROW((void)restore_adamw_state(replace_field(record, "version", std::uint64_t{2}), snapshot),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_adamw_state(replace_field(record, "paths", true), snapshot), std::logic_error);
}

TEST(AdamWStatePersistenceTest, RejectsMismatchedParameterTies)
{
    const auto snapshot = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto record = capture_optimizer_state(snapshot, optimizer.initialize(snapshot));
    const auto &paths = record.get("paths").get<Record>();
    const auto changed_paths = replace_field(paths, "second", paths.get("frozen"));

    EXPECT_THROW((void)restore_adamw_state(replace_field(record, "paths", changed_paths), snapshot),
                 std::invalid_argument);
}

TEST(AdamWStatePersistenceTest, RejectsInvalidMomentEntries)
{
    const auto snapshot = make_snapshot();
    const nn::AdamW optimizer{0.1F, 0.0F};
    const auto record = capture_optimizer_state(snapshot, optimizer.initialize(snapshot));
    const auto &moments = record.get("moments").get<Value::List>();
    ASSERT_EQ(moments.size(), 1U);
    const auto &moment = moments.front().get<Record>();

    const auto with_moments = [&](const Value::List &entries) { return replace_field(record, "moments", entries); };

    EXPECT_THROW((void)restore_adamw_state(with_moments({}), snapshot), std::invalid_argument);
    EXPECT_THROW((void)restore_adamw_state(with_moments({moment, moment}), snapshot), std::invalid_argument);
    EXPECT_THROW(
        (void)restore_adamw_state(with_moments({replace_field(moment, "parameter", std::uint64_t{1})}), snapshot),
        std::invalid_argument);
    EXPECT_THROW(
        (void)restore_adamw_state(with_moments({replace_field(moment, "parameter", std::uint64_t{2})}), snapshot),
        std::out_of_range);
    EXPECT_THROW((void)restore_adamw_state(with_moments({replace_field(moment, "first", from_data({0.0F}, Shape{1}))}),
                                           snapshot),
                 std::invalid_argument);
}
} // namespace minitensor::nn::test
