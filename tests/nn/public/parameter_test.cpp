#include <minitensor/nn/parameter.hpp>

#include <functional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>

#include "nn/support/parameter_trees.hpp"

namespace minitensor::nn::test
{
using support::ParameterPair;

namespace
{
struct EmptyTree
{
    template <typename Self, typename Visitor>
    static void visit_members(Self &, Visitor &)
    {
    }
};
} // namespace

static_assert(!std::is_default_constructible_v<ParameterId>);
static_assert(!std::is_constructible_v<ParameterId, ParameterId::value_type>);
static_assert(!std::is_copy_constructible_v<ParameterBuilder>);

TEST(ParameterIdTest, IsOpaqueComparableAndHashable)
{
    ParameterBuilder builder;
    const ParameterId first = builder.add({full(Shape{2}, 1.0F)});
    const ParameterId copied = first;
    const ParameterId second = builder.add({full(Shape{2}, 1.0F)});

    EXPECT_EQ(copied, first);
    EXPECT_NE(second, first);
    EXPECT_EQ(std::hash<ParameterId>{}(copied), std::hash<ParameterId>{}(first));
    EXPECT_EQ((std::unordered_set<ParameterId>{first, copied, second}).size(), 2U);

    ParameterBuilder other_builder;
    EXPECT_NE(other_builder.add({full(Shape{}, 1.0F)}), first);
}

TEST(ParameterSnapshotTest, ExposesValuesAndMetadataFromTheMap)
{
    ParameterBuilder builder;
    const auto defaults = builder.add({full(Shape{2, 1}, 3.0F)});
    const auto frozen = builder.add({full(Shape{}, 4.0F), {.trainable = false}});
    const auto snapshot = std::move(builder).finish(ParameterPair{defaults, frozen});

    EXPECT_TRUE(snapshot.get(defaults).metadata.trainable);
    EXPECT_EQ(snapshot.value(defaults).shape(), (Shape{2, 1}));
    EXPECT_EQ(snapshot.value(defaults).dtype(), DType::Float32);
    EXPECT_EQ(snapshot.value(defaults).device(), Device::cpu());
    EXPECT_EQ(to_vector(snapshot.value(defaults)), (std::vector<float>{3.0F, 3.0F}));
    EXPECT_FALSE(snapshot.get(frozen).metadata.trainable);
    EXPECT_FLOAT_EQ(item(snapshot.value(frozen)), 4.0F);
    EXPECT_EQ(snapshot.parameters().size(), 2U);
    EXPECT_TRUE(snapshot.parameters().contains(defaults));
}

TEST(ParameterSnapshotTest, CopyAndMovePreserveIdentityAndState)
{
    ParameterBuilder builder;
    const auto id = builder.add({full(Shape{2}, 2.0F), {.trainable = false}});
    const auto original = std::move(builder).finish(ParameterPair{id, id});
    auto copied = original;
    auto moved = std::move(copied);

    EXPECT_EQ(moved.tree().first(), id);
    EXPECT_EQ(moved.tree().second(), id);
    EXPECT_EQ(moved.parameters().size(), 1U);
    EXPECT_FALSE(moved.get(id).metadata.trainable);
    EXPECT_EQ(to_vector(moved.value(id)), (std::vector<float>{2.0F, 2.0F}));
    EXPECT_EQ(to_vector(original.value(id)), to_vector(moved.value(id)));
}

TEST(ParameterSnapshotTest, RejectsUnknownIdsEvenWhenMapSizeMatches)
{
    ParameterBuilder builder;
    const auto known = builder.add({full(Shape{}, 1.0F)});
    ParameterBuilder other;
    const auto unknown = other.add({full(Shape{}, 2.0F)});

    EXPECT_THROW((void)std::move(builder).finish(ParameterPair{known, unknown}), std::invalid_argument);
    EXPECT_THROW((void)ParameterSnapshot(ParameterPair{unknown, unknown},
                                         ParameterMap{ParameterMap::Map{{known, {full(Shape{}, 1.0F)}}}}),
                 std::invalid_argument);
}

TEST(ParameterSnapshotTest, RejectsUnreferencedStates)
{
    ParameterBuilder builder;
    const auto id = builder.add({full(Shape{}, 1.0F)});
    (void)builder.add({full(Shape{}, 2.0F)});
    EXPECT_THROW((void)std::move(builder).finish(ParameterPair{id, id}), std::invalid_argument);
}

TEST(ParameterSnapshotTest, AcceptsEmptyTreesAndRejectsMissingLookups)
{
    ParameterBuilder empty;
    const auto snapshot = std::move(empty).finish(EmptyTree{});
    EXPECT_EQ(snapshot.parameters().size(), 0U);
    ParameterBuilder other;
    const auto unknown = other.add({full(Shape{}, 1.0F)});
    EXPECT_FALSE(snapshot.parameters().contains(unknown));
    EXPECT_THROW((void)snapshot.get(unknown), std::out_of_range);
    EXPECT_THROW((void)snapshot.value(unknown), std::out_of_range);
}

TEST(ParameterSnapshotTest, TransformsEachStateOncePreservingTiesAndSource)
{
    ParameterBuilder builder;
    const auto id = builder.add({full(Shape{2}, 1.0F), {.trainable = false}});
    const auto source = std::move(builder).finish(ParameterPair{id, id});
    std::size_t invocations = 0;
    const auto transformed = transform_parameters(source, [&](ParameterId visited, const ParameterState &state) {
        ++invocations;
        EXPECT_EQ(visited, id);
        return ParameterState{full_like(state.value, 7.0F), state.metadata};
    });

    EXPECT_EQ(invocations, 1U);
    EXPECT_EQ(transformed.tree().first(), id);
    EXPECT_EQ(transformed.tree().second(), id);
    EXPECT_FALSE(transformed.get(id).metadata.trainable);
    EXPECT_EQ(to_vector(transformed.value(id)), (std::vector<float>{7.0F, 7.0F}));
    EXPECT_EQ(to_vector(source.value(id)), (std::vector<float>{1.0F, 1.0F}));
}

TEST(ParameterSnapshotTest, TransformsMetadataIndependentlyOfValues)
{
    ParameterBuilder builder;
    const auto id = builder.add({full(Shape{2}, 6.0F)});
    const auto original = std::move(builder).finish(ParameterPair{id, id});
    const auto updated = transform_parameters(original, [](ParameterId, const ParameterState &state) {
        return ParameterState{state.value, {.trainable = false}};
    });

    EXPECT_EQ(updated.tree().first(), id);
    EXPECT_FALSE(updated.get(id).metadata.trainable);
    EXPECT_TRUE(original.get(id).metadata.trainable);
    EXPECT_EQ(to_vector(updated.value(id)), (std::vector<float>{6.0F, 6.0F}));
}
} // namespace minitensor::nn::test
