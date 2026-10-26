#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/nn/module.hpp>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/random.hpp>

namespace minitensor::nn::test
{
namespace
{
class TiedGradientTree final
{
  public:
    TiedGradientTree(ParameterId shared, ParameterId frozen)
        : first_{shared}, second_{std::move(shared)}, frozen_{std::move(frozen)}
    {
    }

    [[nodiscard]] const ParameterId &first() const noexcept
    {
        return first_;
    }

    [[nodiscard]] const ParameterId &second() const noexcept
    {
        return second_;
    }

    [[nodiscard]] const ParameterId &frozen() const noexcept
    {
        return frozen_;
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter("first", self.first_);
        visitor.parameter("second", self.second_);
        visitor.parameter("frozen", self.frozen_);
    }

    ParameterId first_;
    ParameterId second_;
    ParameterId frozen_;

    friend class ::minitensor::nn::ParameterTreeAccess;
};
} // namespace

TEST(ParameterGradientsTest, AssociatesGradientsWithParameterIds)
{
    const Linear linear{2, 1};
    const auto parameters = initialize(linear, RandomKey{41});
    ASSERT_NE(parameters.tree().bias(), nullptr);

    const Tensor loss =
        sum(parameters.value(parameters.tree().weight()) * parameters.value(parameters.tree().weight())) +
        sum(parameters.value(*parameters.tree().bias()) * parameters.value(*parameters.tree().bias()));
    const ParameterGradients gradients = parameter_gradients(loss, parameters);

    ASSERT_EQ(gradients.size(), 2);
    ASSERT_TRUE(gradients.contains(parameters.tree().weight()));
    ASSERT_TRUE(gradients.contains(*parameters.tree().bias()));

    const std::vector<float> weight = to_vector(parameters.value(parameters.tree().weight()));
    const std::vector<float> weight_gradient = to_vector(gradients.at(parameters.tree().weight()));
    ASSERT_EQ(weight_gradient.size(), weight.size());
    for (std::size_t index = 0; index < weight.size(); ++index)
    {
        EXPECT_FLOAT_EQ(weight_gradient[index], 2.0F * weight[index]);
    }

    const std::vector<float> bias = to_vector(parameters.value(*parameters.tree().bias()));
    const std::vector<float> bias_gradient = to_vector(gradients.at(*parameters.tree().bias()));
    ASSERT_EQ(bias_gradient.size(), bias.size());
    for (std::size_t index = 0; index < bias.size(); ++index)
    {
        EXPECT_FLOAT_EQ(bias_gradient[index], 2.0F * bias[index]);
    }
}

TEST(ParameterGradientsTest, FiltersFrozenAndDeduplicatesTiedParameters)
{
    ParameterBuilder builder;
    const ParameterId shared = builder.add(ParameterState{from_data({1.0F, 2.0F}, Shape{2})});
    const ParameterId frozen =
        builder.add(ParameterState{from_data({3.0F, 4.0F}, Shape{2}), ParameterMetadata{.trainable = false}});
    const auto tree = std::move(builder).finish(TiedGradientTree{shared, frozen});
    const Tensor loss =
        sum(tree.value(tree.tree().first()) + tree.value(tree.tree().second()) + tree.value(tree.tree().frozen()));

    const ParameterGradients gradients = parameter_gradients(loss, tree);

    ASSERT_EQ(gradients.size(), 1);
    ASSERT_TRUE(gradients.contains(shared));
    EXPECT_FALSE(gradients.contains(frozen));
    EXPECT_EQ(to_vector(gradients.at(shared)), (std::vector{2.0F, 2.0F}));
}
} // namespace minitensor::nn::test
