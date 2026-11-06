#include <minitensor/persistence/parameters.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/evaluation.hpp>
#include <minitensor/ops/creation.hpp>

#include "nn/support/parameter_trees.hpp"
#include "persistence/support/record.hpp"

namespace minitensor::persistence::test
{
using nn::test::support::EmptyTree;
using nn::test::support::MixedParameterTree;
using nn::test::support::ParameterPair;
using support::replace_field;

namespace
{

struct NestedTree
{
    ParameterPair layer;

    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.child("layer", self.layer);
    }
};

nn::ParameterSnapshot<ParameterPair> make_pair(nn::ParameterState first = {full(Shape{2}, 1.0F)},
                                               nn::ParameterState second = {full(Shape{2}, 2.0F)})
{
    nn::ParameterBuilder builder;
    const auto first_id = builder.add(std::move(first));
    const auto second_id = builder.add(std::move(second));
    return std::move(builder).finish(ParameterPair{first_id, second_id});
}

} // namespace

TEST(ParameterPersistenceTest, CapturesAllPathsAndEachUniqueStateWithMetadata)
{
    nn::ParameterBuilder builder;
    const auto shared = builder.add({full(Shape{2}, 3.0F)});
    const auto frozen = builder.add({full(Shape{}, 7.0F), {.trainable = false}});
    const auto snapshot = std::move(builder).finish(MixedParameterTree{shared, frozen});

    const auto record = capture_parameters(snapshot);
    EXPECT_EQ(record.get("kind").get<std::string>(), "parameter_snapshot");
    EXPECT_EQ(record.get("version").get<std::uint64_t>(), 1U);
    const auto &paths = record.get("paths").get<Record>();
    const auto &states = record.get("states").get<Value::List>();
    EXPECT_EQ(paths.size(), 3U);
    EXPECT_EQ(paths.get("first").get<std::uint64_t>(), 0U);
    EXPECT_EQ(paths.get("second").get<std::uint64_t>(), 0U);
    EXPECT_EQ(paths.get("frozen").get<std::uint64_t>(), 1U);
    ASSERT_EQ(states.size(), 2U);
    const auto &first = states[0].get<Record>();
    const auto &second = states[1].get<Record>();
    EXPECT_EQ(to_vector(first.get("value").get<Tensor>()), (std::vector<float>{3.0F, 3.0F}));
    EXPECT_TRUE(first.get("trainability").get<bool>());
    EXPECT_FLOAT_EQ(item(second.get("value").get<Tensor>()), 7.0F);
    EXPECT_FALSE(second.get("trainability").get<bool>());
}

TEST(ParameterPersistenceTest, RestoresValuesAndMetadataPreservingTargetIdsAndOriginals)
{
    const auto source = make_pair({full(Shape{2}, 9.0F), {.trainable = false}}, {full(Shape{}, 4.0F)});
    auto target = make_pair({full(Shape{2}, -1.0F)}, {full(Shape{}, -2.0F), {.trainable = false}});
    const auto first = target.tree().first();
    const auto second = target.tree().second();
    ASSERT_NE(first, source.tree().first());
    ASSERT_NE(second, source.tree().second());

    // The temporary captured record is destroyed before the result is inspected.
    const auto restored = restore_parameters(capture_parameters(source), target);
    EXPECT_EQ(restored.tree().first(), first);
    EXPECT_EQ(restored.tree().second(), second);
    EXPECT_EQ(restored.parameters().size(), 2U);
    EXPECT_EQ(to_vector(restored.value(first)), (std::vector<float>{9.0F, 9.0F}));
    EXPECT_FLOAT_EQ(item(restored.value(second)), 4.0F);
    EXPECT_FALSE(restored.get(first).metadata.trainable);
    EXPECT_TRUE(restored.get(second).metadata.trainable);
    EXPECT_EQ(to_vector(target.value(first)), (std::vector<float>{-1.0F, -1.0F}));
    EXPECT_FLOAT_EQ(item(target.value(second)), -2.0F);
    EXPECT_TRUE(target.get(first).metadata.trainable);
    EXPECT_FALSE(target.get(second).metadata.trainable);
    EXPECT_EQ(to_vector(source.value(source.tree().first())), (std::vector<float>{9.0F, 9.0F}));

    const auto consumed = restore_parameters(capture_parameters(source), std::move(target));
    EXPECT_EQ(consumed.tree().first(), first);
    EXPECT_EQ(consumed.tree().second(), second);
    EXPECT_EQ(to_vector(consumed.value(first)), to_vector(restored.value(first)));
}

TEST(ParameterPersistenceTest, RoundTripsNestedPathsAndTiedParameters)
{
    nn::ParameterBuilder source_builder;
    const auto source_id = source_builder.add({full(Shape{2}, 5.0F), {.trainable = false}});
    const auto source = std::move(source_builder).finish(NestedTree{ParameterPair{source_id, source_id}});
    nn::ParameterBuilder target_builder;
    const auto target_id = target_builder.add({full(Shape{2}, 0.0F)});
    const auto target = std::move(target_builder).finish(NestedTree{ParameterPair{target_id, target_id}});

    const auto record = capture_parameters(source);
    const auto &paths = record.get("paths").get<Record>();
    EXPECT_EQ(paths.get("layer.first").get<std::uint64_t>(), 0U);
    EXPECT_EQ(paths.get("layer.second").get<std::uint64_t>(), 0U);
    const auto restored = restore_parameters(record, target);
    EXPECT_EQ(restored.tree().layer.first(), target_id);
    EXPECT_EQ(restored.tree().layer.second(), target_id);
    EXPECT_EQ(restored.parameters().size(), 1U);
    EXPECT_FALSE(restored.get(target_id).metadata.trainable);
    EXPECT_EQ(to_vector(restored.value(target_id)), (std::vector<float>{5.0F, 5.0F}));
}

TEST(ParameterPersistenceTest, RoundTripsEmptySnapshots)
{
    nn::ParameterBuilder builder;
    const auto target = std::move(builder).finish(EmptyTree{});
    const auto record = capture_parameters(target);
    EXPECT_TRUE(record.get("paths").get<Record>().empty());
    EXPECT_TRUE(record.get("states").get<Value::List>().empty());
    const auto restored = restore_parameters(record, target);
    EXPECT_EQ(restored.parameters().size(), 0U);
}

TEST(ParameterPersistenceTest, CapturesAndRestoresWithoutEvaluatingTensors)
{
    const TensorOptions options{DType::Float32, Device::cpu(1)};
    const auto source = make_pair({full(Shape{2}, 3.0F, options)}, {full(Shape{2}, 4.0F, options)});
    const auto target = make_pair({full(Shape{2}, 0.0F, options)}, {full(Shape{2}, 0.0F, options)});
    const auto restored = restore_parameters(capture_parameters(source), target);

    EXPECT_EQ(restored.value(target.tree().first()).shape(), (Shape{2}));
    EXPECT_EQ(restored.value(target.tree().first()).device(), Device::cpu(1));
    EXPECT_EQ(restored.value(target.tree().second()).device(), Device::cpu(1));
    // This unavailable device would have caused capture or restore to throw if either evaluated.
    EXPECT_THROW(eval(restored.value(target.tree().first())), std::runtime_error);
}

TEST(ParameterPersistenceTest, RejectsInvalidSchemaAndFieldTypes)
{
    const auto target = make_pair();
    const auto record = capture_parameters(target);
    EXPECT_THROW((void)restore_parameters(replace_field(record, "kind", "optimizer"), target), std::invalid_argument);
    EXPECT_THROW((void)restore_parameters(replace_field(record, "version", std::uint64_t{2}), target),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_parameters(replace_field(record, "version", "1"), target), std::logic_error);

    Record missing;
    for (const auto &[name, value] : record)
    {
        if (name != "version")
        {
            missing.add(name, value);
        }
    }
    EXPECT_THROW((void)restore_parameters(missing, target), std::invalid_argument);
    auto extra = record;
    extra.add("unexpected", true);
    EXPECT_THROW((void)restore_parameters(extra, target), std::invalid_argument);

    auto states = record.get("states").get<Value::List>();
    states[0] = replace_field(states[0].get<Record>(), "trainability", "true");
    EXPECT_THROW((void)restore_parameters(replace_field(record, "states", states), target), std::logic_error);
    states[0] = true;
    EXPECT_THROW((void)restore_parameters(replace_field(record, "states", states), target), std::logic_error);
}

TEST(ParameterPersistenceTest, RejectsInvalidStateReferencesAndUnreferencedStates)
{
    const auto target = make_pair();
    const auto record = capture_parameters(target);
    const auto &paths = record.get("paths").get<Record>();
    const auto bad_index = replace_field(paths, "first", std::numeric_limits<std::uint64_t>::max());
    EXPECT_THROW((void)restore_parameters(replace_field(record, "paths", bad_index), target), std::out_of_range);
    const auto wrong_type = replace_field(paths, "first", std::int64_t{0});
    EXPECT_THROW((void)restore_parameters(replace_field(record, "paths", wrong_type), target), std::logic_error);

    auto states = record.get("states").get<Value::List>();
    const auto unused = states.front();
    states.push_back(unused);
    EXPECT_THROW((void)restore_parameters(replace_field(record, "states", states), target), std::invalid_argument);
}

TEST(ParameterPersistenceTest, RejectsMissingAndUnexpectedPaths)
{
    const auto target = make_pair();
    const auto record = capture_parameters(target);
    Record missing_path;
    missing_path.add("first", std::uint64_t{0});
    // Keep the saved representation internally valid so this reaches target path matching.
    const Value::List single_state{record.get("states").get<Value::List>()[0]};
    const auto missing = replace_field(replace_field(record, "states", single_state), "paths", missing_path);
    EXPECT_THROW((void)restore_parameters(missing, target), std::invalid_argument);

    auto extra_paths = record.get("paths").get<Record>();
    extra_paths.add("unexpected", std::uint64_t{0});
    EXPECT_THROW((void)restore_parameters(replace_field(record, "paths", extra_paths), target), std::invalid_argument);
}

TEST(ParameterPersistenceTest, RejectsSharingMismatchesInBothDirections)
{
    const auto distinct = make_pair();
    nn::ParameterBuilder builder;
    const auto shared = builder.add({full(Shape{2}, 0.0F)});
    const auto tied = std::move(builder).finish(ParameterPair{shared, shared});

    EXPECT_THROW((void)restore_parameters(capture_parameters(tied), distinct), std::invalid_argument);
    EXPECT_THROW((void)restore_parameters(capture_parameters(distinct), tied), std::invalid_argument);
    EXPECT_EQ(tied.tree().first(), tied.tree().second());
    EXPECT_NE(distinct.tree().first(), distinct.tree().second());
}

TEST(ParameterPersistenceTest, RejectsSpecificationMismatchesWithoutChangingTarget)
{
    const auto target = make_pair();
    const auto first = target.tree().first();
    const auto second = target.tree().second();
    for (const Tensor &incompatible : {full(Shape{1, 2}, 3.0F), full(Shape{2}, 3, TensorOptions{DType::Int64}),
                                       full(Shape{2}, 3.0F, TensorOptions{DType::Float32, Device::cpu(1)})})
    {
        const auto source = target.transform([&](nn::ParameterId id, const nn::ParameterState &state) {
            return nn::ParameterState{id == first ? incompatible : state.value, state.metadata};
        });
        EXPECT_THROW((void)restore_parameters(capture_parameters(source), target), std::invalid_argument);
    }
    EXPECT_EQ(to_vector(target.value(first)), (std::vector<float>{1.0F, 1.0F}));
    EXPECT_EQ(to_vector(target.value(second)), (std::vector<float>{2.0F, 2.0F}));
}
} // namespace minitensor::persistence::test
