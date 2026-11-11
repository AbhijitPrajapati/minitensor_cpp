#include <minitensor/nn/persistence.hpp>

#include <cstdint>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/evaluation.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/optimizers/rmsprop.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/persistence/record.hpp>

#include "nn/support/parameter_trees.hpp"
#include "persistence/support/record.hpp"

namespace minitensor::nn::test
{
using persistence::Record;
using persistence::Value;
using persistence::test::support::replace_field;
using support::MixedParameterTree;
using support::ParameterPair;

namespace
{
ParameterSnapshot<MixedParameterTree> make_snapshot(bool trainable = true, TensorOptions options = {})
{
    ParameterBuilder builder;
    const auto shared = builder.add({full(Shape{}, 1.0F, options), {.trainable = trainable}});
    const auto frozen = builder.add({full(Shape{}, 8.0F, options), {.trainable = !trainable}});
    return std::move(builder).finish(MixedParameterTree{shared, frozen});
}
} // namespace

TEST(RMSPropCheckpointTest, ResumesWithTiedParametersFreshIdsAndSavedTrainability)
{
    const auto source = make_snapshot();
    const RMSProp optimizer{0.1F, 0.75F, 0.25F};
    const ParameterGradients first_gradients{{source.tree().first(), from_data({2.0F}, Shape{})}};
    const auto [updated, state] = optimizer.step(source, optimizer.initialize(source), first_gradients);

    std::stringstream archive(std::ios::in | std::ios::out | std::ios::binary);
    capture_checkpoint(updated, optimizer, state).write(archive);
    archive.seekg(0);
    const auto record = Record::read(archive);
    EXPECT_TRUE(record.contains({"kind", "version", "paths", "states", "rmsprop"}, true));
    const auto &payload = record.get("rmsprop").get<Record>();
    EXPECT_TRUE(payload.contains({"learning_rate", "alpha", "epsilon", "moving_averages"}, true));

    const auto target = make_snapshot(false);
    const auto target_id = target.tree().first();
    ASSERT_NE(target_id, source.tree().first());
    const auto [restored, restored_optimizer, restored_state] = restore_checkpoint<RMSProp>(record, target);
    EXPECT_EQ(restored.tree().first(), target_id);
    EXPECT_EQ(restored.tree().second(), target_id);
    EXPECT_TRUE(restored.get(target_id).metadata.trainable);
    EXPECT_FALSE(restored.get(target.tree().frozen()).metadata.trainable);
    EXPECT_FLOAT_EQ(item(restored.value(target_id)), item(updated.value(source.tree().first())));
    EXPECT_FLOAT_EQ(restored_optimizer.learning_rate(), optimizer.learning_rate());
    EXPECT_FLOAT_EQ(restored_optimizer.alpha(), optimizer.alpha());
    EXPECT_FLOAT_EQ(restored_optimizer.epsilon(), optimizer.epsilon());

    const auto gradient = from_data({1.0F}, Shape{});
    const auto [expected, expected_state] = optimizer.step(updated, state, {{source.tree().first(), gradient}});
    const auto [actual, actual_state] = restored_optimizer.step(restored, restored_state, {{target_id, gradient}});
    EXPECT_FLOAT_EQ(item(actual.value(target_id)), item(expected.value(source.tree().first())));
    EXPECT_FLOAT_EQ(item(actual.value(target.tree().frozen())), 8.0F);
    EXPECT_FALSE(target.get(target_id).metadata.trainable);
    EXPECT_FLOAT_EQ(item(target.value(target_id)), 1.0F);
}

TEST(RMSPropCheckpointTest, PreservesDistinctParameterAverages)
{
    ParameterBuilder source_builder;
    const auto first = source_builder.add({from_data({1.0F}, Shape{})});
    const auto second = source_builder.add({from_data({-2.0F}, Shape{})});
    const auto source = std::move(source_builder).finish(ParameterPair{first, second});
    const RMSProp optimizer{0.1F, 0.75F, 0.25F};
    const auto [updated, state] =
        optimizer.step(source, optimizer.initialize(source),
                       {{first, from_data({2.0F}, Shape{})}, {second, from_data({4.0F}, Shape{})}});
    const auto record = capture_checkpoint(updated, optimizer, state);

    ParameterBuilder target_builder;
    const auto target_first = target_builder.add({from_data({0.0F}, Shape{})});
    const auto target_second = target_builder.add({from_data({0.0F}, Shape{})});
    auto target = std::move(target_builder).finish(ParameterPair{target_first, target_second});
    const auto restored = restore_checkpoint<RMSProp>(record, std::move(target));
    const auto first_gradient = from_data({1.0F}, Shape{});
    const auto second_gradient = from_data({-2.0F}, Shape{});
    const auto [expected, expected_state] =
        optimizer.step(updated, state, {{first, first_gradient}, {second, second_gradient}});
    const auto [actual, actual_state] =
        restored.optimizer.step(restored.snapshot, restored.optimizer_state,
                                {{target_first, first_gradient}, {target_second, second_gradient}});

    EXPECT_FLOAT_EQ(item(actual.value(target_first)), item(expected.value(first)));
    EXPECT_FLOAT_EQ(item(actual.value(target_second)), item(expected.value(second)));
}

TEST(RMSPropCheckpointTest, RoundTripsStateWhenFrozenParametersComeFirst)
{
    const auto source = make_snapshot(false);
    const RMSProp optimizer{0.1F};
    const auto record = capture_checkpoint(source, optimizer, optimizer.initialize(source));
    const auto &paths = record.get("paths").get<Record>();
    const auto &averages = record.get("rmsprop").get<Record>().get("moving_averages").get<Value::List>();
    ASSERT_EQ(averages.size(), 1U);
    EXPECT_EQ(paths.get("first").get<std::uint64_t>(), paths.get("second").get<std::uint64_t>());
    EXPECT_EQ(averages.front().get<Record>().get("parameter").get<std::uint64_t>(),
              paths.get("frozen").get<std::uint64_t>());
    const auto restored = restore_checkpoint<RMSProp>(record, make_snapshot());
    EXPECT_FALSE(restored.snapshot.get(restored.snapshot.tree().first()).metadata.trainable);
    EXPECT_TRUE(restored.snapshot.get(restored.snapshot.tree().frozen()).metadata.trainable);
}

TEST(RMSPropCheckpointTest, RoundTripsEmptyAndFrozenSnapshots)
{
    const RMSProp optimizer{0.1F};
    const auto empty = ParameterBuilder{}.finish(support::EmptyTree{});
    const auto empty_record = capture_checkpoint(empty, optimizer, optimizer.initialize(empty));
    const auto restored_empty = restore_checkpoint<RMSProp>(empty_record, empty);
    EXPECT_EQ(restored_empty.snapshot.parameters().size(), 0U);

    ParameterBuilder builder;
    const auto first = builder.add({full(Shape{2}, 1, TensorOptions{DType::Int64}), {.trainable = false}});
    const auto second = builder.add({full(Shape{}, 3, TensorOptions{DType::Int64}), {.trainable = false}});
    const auto frozen = std::move(builder).finish(ParameterPair{first, second});
    const auto record = capture_checkpoint(frozen, optimizer, optimizer.initialize(frozen));
    EXPECT_TRUE(record.get("rmsprop").get<Record>().get("moving_averages").get<Value::List>().empty());
    const auto restored = restore_checkpoint<RMSProp>(record, frozen);
    EXPECT_EQ(to_vector<std::int64_t>(restored.snapshot.value(first)), (std::vector<std::int64_t>{1, 1}));
    EXPECT_EQ(item<std::int64_t>(restored.snapshot.value(second)), 3);
    EXPECT_FALSE(restored.snapshot.get(first).metadata.trainable);
}

TEST(RMSPropCheckpointTest, CapturesAndRestoresWithoutEvaluatingTensors)
{
    const TensorOptions options{DType::Float32, Device::cpu(1)};
    const auto source = make_snapshot(true, options);
    const RMSProp optimizer{0.1F};
    const auto restored = restore_checkpoint<RMSProp>(
        capture_checkpoint(source, optimizer, optimizer.initialize(source)), make_snapshot(false, options));
    EXPECT_EQ(restored.snapshot.value(restored.snapshot.tree().first()).device(), Device::cpu(1));
    const auto record = capture_checkpoint(restored.snapshot, restored.optimizer, restored.optimizer_state);
    const auto &average = record.get("rmsprop")
                              .get<Record>()
                              .get("moving_averages")
                              .get<Value::List>()
                              .front()
                              .get<Record>()
                              .get("average")
                              .get<Tensor>();
    EXPECT_EQ(average.device(), Device::cpu(1));
    EXPECT_THROW(eval(average), std::runtime_error);
}

TEST(RMSPropCheckpointTest, RejectsStateForDifferentParametersOrTrainability)
{
    const auto source = make_snapshot();
    const RMSProp optimizer{0.1F};
    const auto state = optimizer.initialize(source);
    EXPECT_THROW((void)capture_checkpoint(make_snapshot(), optimizer, state), std::invalid_argument);
    const auto changed = source.transform([](ParameterId, const ParameterState &parameter) {
        return ParameterState{parameter.value, {.trainable = !parameter.metadata.trainable}};
    });
    EXPECT_THROW((void)capture_checkpoint(changed, optimizer, state), std::invalid_argument);
    const auto record = capture_checkpoint(source, optimizer, state);
    EXPECT_NO_THROW((void)restore_checkpoint<RMSProp>(record, make_snapshot(false)));
}

TEST(RMSPropCheckpointTest, RejectsInvalidCheckpointAndPayloadFields)
{
    const auto source = make_snapshot();
    const RMSProp optimizer{0.1F};
    const auto record = capture_checkpoint(source, optimizer, optimizer.initialize(source));
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(replace_field(record, "kind", "snapshot"), source),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(replace_field(record, "version", std::uint64_t{2}), source),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(replace_field(record, "rmsprop", true), source), std::logic_error);
    auto extra = record;
    extra.add("unexpected", true);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(extra, source), std::invalid_argument);

    const auto &payload = record.get("rmsprop").get<Record>();
    Record missing;
    for (const auto &[name, value] : payload)
    {
        if (name != "alpha")
        {
            missing.add(name, value);
        }
    }
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(replace_field(record, "rmsprop", missing), source),
                 std::invalid_argument);
    auto extra_payload = payload;
    extra_payload.add("unexpected", true);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(replace_field(record, "rmsprop", extra_payload), source),
                 std::invalid_argument);
}

TEST(RMSPropCheckpointTest, RejectsInvalidConfigurationFields)
{
    const auto source = make_snapshot();
    const RMSProp optimizer{0.1F};
    const auto record = capture_checkpoint(source, optimizer, optimizer.initialize(source));
    const auto &payload = record.get("rmsprop").get<Record>();
    const auto with_field = [&](std::string_view field, const Value &value) {
        return replace_field(record, "rmsprop", replace_field(payload, field, value));
    };
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_field("learning_rate", "0.1"), source), std::logic_error);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_field("learning_rate", -0.1F), source), std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_field("alpha", 1.0F), source), std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_field("epsilon", 0.0F), source), std::invalid_argument);
}

TEST(RMSPropCheckpointTest, RejectsInvalidAverageReferencesAndFields)
{
    const auto source = make_snapshot();
    const RMSProp optimizer{0.1F};
    const auto record = capture_checkpoint(source, optimizer, optimizer.initialize(source));
    const auto &payload = record.get("rmsprop").get<Record>();
    const auto &averages = payload.get("moving_averages").get<Value::List>();
    ASSERT_EQ(averages.size(), 1U);
    const auto &average = averages.front().get<Record>();
    const auto with_averages = [&](const Value::List &entries) {
        return replace_field(record, "rmsprop", replace_field(payload, "moving_averages", entries));
    };
    const auto &paths = record.get("paths").get<Record>();
    const auto out_of_range = static_cast<std::uint64_t>(record.get("states").get<Value::List>().size());
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_averages({}), source), std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_averages({average, average}), source), std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(
                     with_averages({replace_field(average, "parameter", paths.get("frozen"))}), source),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)restore_checkpoint<RMSProp>(with_averages({replace_field(average, "parameter", out_of_range)}), source),
        std::out_of_range);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(
                     with_averages({replace_field(average, "parameter", std::int64_t{0})}), source),
                 std::logic_error);
    Record missing;
    missing.add("parameter", std::uint64_t{0});
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_averages({missing}), source), std::invalid_argument);
    auto extra = average;
    extra.add("unexpected", true);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_averages({extra}), source), std::invalid_argument);
    EXPECT_THROW((void)restore_checkpoint<RMSProp>(with_averages({true}), source), std::logic_error);
}

TEST(RMSPropCheckpointTest, RejectsAverageSpecificationMismatchesWithoutChangingTarget)
{
    const auto source = make_snapshot();
    const RMSProp optimizer{0.1F};
    const auto record = capture_checkpoint(source, optimizer, optimizer.initialize(source));
    const auto &payload = record.get("rmsprop").get<Record>();
    const auto &average = payload.get("moving_averages").get<Value::List>().front().get<Record>();
    for (const Tensor &invalid : {full(Shape{1}, 0.0F), full(Shape{}, 0, TensorOptions{DType::Int64}),
                                  full(Shape{}, 0.0F, TensorOptions{DType::Float32, Device::cpu(1)})})
    {
        const Value::List entries{replace_field(average, "average", invalid)};
        const auto changed = replace_field(record, "rmsprop", replace_field(payload, "moving_averages", entries));
        EXPECT_THROW((void)restore_checkpoint<RMSProp>(changed, source), std::invalid_argument);
    }
    EXPECT_FLOAT_EQ(item(source.value(source.tree().first())), 1.0F);
    EXPECT_TRUE(source.get(source.tree().first()).metadata.trainable);
}
} // namespace minitensor::nn::test
