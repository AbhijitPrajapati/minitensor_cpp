#include <minitensor/nn/leafification.hpp>

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>

#include "nn/support/parameter_trees.hpp"

namespace minitensor::nn::test
{
using support::TiedParameterTree;

TEST(ParameterLeafificationTest, PreservesParameterStateAndTies)
{
    ParameterBuilder builder;
    const ParameterId shared = builder.add(ParameterState{full(Shape{2}, 4.0F), ParameterMetadata{.trainable = false}});
    const auto source = std::move(builder).finish(TiedParameterTree{shared});

    const auto leafified = leafify_parameter_values(source);

    EXPECT_EQ(leafified.tree().first(), source.tree().first());
    EXPECT_EQ(leafified.tree().second(), source.tree().second());
    EXPECT_EQ(leafified.get(leafified.tree().first()).metadata, source.get(source.tree().first()).metadata);
    EXPECT_EQ(leafified.get(leafified.tree().second()).metadata, source.get(source.tree().second()).metadata);
    EXPECT_EQ(to_vector(leafified.value(leafified.tree().first())), (std::vector<float>{4.0F, 4.0F}));
    EXPECT_EQ(to_vector(leafified.value(leafified.tree().second())), (std::vector<float>{4.0F, 4.0F}));
    EXPECT_EQ(to_vector(source.value(source.tree().first())), (std::vector<float>{4.0F, 4.0F}));
}

TEST(ParameterLeafificationTest, DetachesThePreviousGraphWhilePreservingTiedTensorIdentity)
{
    const Tensor input = full(Shape{2}, 2.0F);
    ParameterBuilder builder;
    const auto id = builder.add({input * input});
    const auto source = std::move(builder).finish(TiedParameterTree{id});
    const auto leafified = leafify_parameter_values(source);

    const Tensor before = sum(source.value(id));
    const Tensor after = sum(leafified.value(leafified.tree().first()) + leafified.value(leafified.tree().second()));
    EXPECT_EQ(to_vector(grad(before, input).front()), (std::vector<float>{4.0F, 4.0F}));
    EXPECT_EQ(to_vector(grad(after, input).front()), (std::vector<float>{0.0F, 0.0F}));
    EXPECT_EQ(to_vector(grad(after, leafified.value(id)).front()), (std::vector<float>{2.0F, 2.0F}));
}
} // namespace minitensor::nn::test
