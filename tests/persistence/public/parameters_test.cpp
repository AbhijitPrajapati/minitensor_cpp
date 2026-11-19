#include <minitensor/persistence/parameters.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/persistence/detail/archive.hpp>

#include "nn/support/parameter_trees.hpp"

namespace minitensor::persistence::test
{
namespace
{
using nn::test::support::EmptyTree;
using nn::test::support::ParameterPair;

auto make_pair(float first, float second, bool shared = false, bool trainable = true)
{
    nn::ParameterBuilder builder;
    const auto first_id = builder.add({full(Shape{}, first), {.trainable = trainable}});
    const auto second_id = shared ? first_id : builder.add({full(Shape{}, second), {.trainable = !trainable}});
    return std::move(builder).finish(ParameterPair{first_id, second_id});
}

struct ReversePair
{
    nn::ParameterId first;
    nn::ParameterId second;

    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter("second", self.second);
        visitor.parameter("first", self.first);
    }
};
} // namespace

TEST(ParameterPersistenceTest, ReplacesValuesWhileRetainingTargetIdsAndMetadata)
{
    const auto source = make_pair(3.0F, 4.0F, false, false);
    const auto target = make_pair(1.0F, 2.0F);
    std::stringstream stream;
    write_parameter_values(stream, source);
    const auto loaded = read_parameter_values(stream, target);

    EXPECT_NE(source.tree().first(), target.tree().first());
    EXPECT_EQ(loaded.tree().first(), target.tree().first());
    EXPECT_EQ(loaded.tree().second(), target.tree().second());
    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().first())), 3.0F);
    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().second())), 4.0F);
    EXPECT_TRUE(loaded.get(loaded.tree().first()).metadata.trainable);
    EXPECT_FALSE(loaded.get(loaded.tree().second()).metadata.trainable);
    EXPECT_FLOAT_EQ(item(target.value(target.tree().first())), 1.0F);
    EXPECT_FLOAT_EQ(item(target.value(target.tree().second())), 2.0F);
    EXPECT_FLOAT_EQ(item(source.value(source.tree().first())), 3.0F);
}

TEST(ParameterPersistenceTest, PreservesTiedParameterIdentity)
{
    const auto source = make_pair(8.0F, 0.0F, true);
    const auto target = make_pair(1.0F, 0.0F, true);
    std::stringstream stream;
    write_parameter_values(stream, source);
    const auto loaded = read_parameter_values(stream, target);

    EXPECT_EQ(loaded.parameters().size(), 1U);
    EXPECT_EQ(loaded.tree().first(), target.tree().first());
    EXPECT_EQ(loaded.tree().first(), loaded.tree().second());
    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().second())), 8.0F);
}

TEST(ParameterPersistenceTest, KeepsDistinctIdsWhenTheirTensorHandleIsShared)
{
    nn::ParameterBuilder builder;
    const Tensor value = full(Shape{}, 7.0F);
    const auto first = builder.add({value});
    const auto second = builder.add({value});
    const auto source = std::move(builder).finish(ParameterPair{first, second});
    std::stringstream stream;
    write_parameter_values(stream, source);
    const auto loaded = read_parameter_values(stream, make_pair(1.0F, 2.0F));

    EXPECT_EQ(loaded.parameters().size(), 2U);
    EXPECT_NE(loaded.tree().first(), loaded.tree().second());
    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().first())), 7.0F);
    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().second())), 7.0F);
}

TEST(ParameterPersistenceTest, MatchesPathsRegardlessOfTraversalOrder)
{
    nn::ParameterBuilder builder;
    const auto first = builder.add({full(Shape{}, 3.0F)});
    const auto second = builder.add({full(Shape{}, 4.0F)});
    const auto source = std::move(builder).finish(ReversePair{first, second});
    std::stringstream stream;
    write_parameter_values(stream, source);
    const auto loaded = read_parameter_values(stream, make_pair(1.0F, 2.0F));

    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().first())), 3.0F);
    EXPECT_FLOAT_EQ(item(loaded.value(loaded.tree().second())), 4.0F);
}

TEST(ParameterPersistenceTest, RejectsBothDirectionsOfSharingMismatch)
{
    for (const bool shared : {false, true})
    {
        SCOPED_TRACE(shared);
        std::stringstream stream;
        write_parameter_values(stream, make_pair(3.0F, 4.0F, shared));
        EXPECT_THROW((void)read_parameter_values(stream, make_pair(1.0F, 2.0F, !shared)), std::invalid_argument);
    }
}

TEST(ParameterPersistenceTest, RejectsMissingAndUnexpectedPaths)
{
    const auto target = make_pair(1.0F, 2.0F);
    const Tensor value = full(Shape{}, 3.0F);
    for (const auto &bindings : std::vector<decltype(detail::ArchiveData::bindings)>{
             {{"first", 0}}, {{"first", 0}, {"second", 1}, {"extra", 0}}, {{"first", 0}, {"other", 1}}})
    {
        std::stringstream stream;
        detail::write_archive(stream, {{value, value}, bindings});
        EXPECT_THROW((void)read_parameter_values(stream, target), std::invalid_argument);
    }
}

TEST(ParameterPersistenceTest, RejectsUnboundTensorRecords)
{
    const Tensor value = full(Shape{}, 3.0F);
    std::stringstream stream;
    detail::write_archive(stream, {{value, value, value}, {{"first", 0}, {"second", 1}}});
    EXPECT_THROW((void)read_parameter_values(stream, make_pair(1.0F, 2.0F)), std::invalid_argument);
}

TEST(ParameterPersistenceTest, ValidatesAllSpecificationsBeforeReplacingTargetValues)
{
    const auto target = make_pair(1.0F, 2.0F);
    for (const Tensor &mismatch : {full(Shape{2}, 4.0F), from_data<std::int64_t>({4}, Shape{})})
    {
        std::stringstream stream;
        detail::write_archive(stream, {{full(Shape{}, 3.0F), mismatch}, {{"first", 0}, {"second", 1}}});
        EXPECT_THROW((void)read_parameter_values(stream, target), std::invalid_argument);
        EXPECT_FLOAT_EQ(item(target.value(target.tree().first())), 1.0F);
        EXPECT_FLOAT_EQ(item(target.value(target.tree().second())), 2.0F);
    }
}

TEST(ParameterPersistenceTest, SupportsEmptySnapshots)
{
    nn::ParameterBuilder builder;
    const auto empty = std::move(builder).finish(EmptyTree{});
    std::stringstream stream;
    write_parameter_values(stream, empty);
    EXPECT_EQ(read_parameter_values(stream, empty).parameters().size(), 0U);
}
} // namespace minitensor::persistence::test
