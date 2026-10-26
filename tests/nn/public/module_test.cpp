#include <minitensor/nn/module.hpp>

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/leafification.hpp>
#include <minitensor/nn/loss.hpp>
#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/optimizers/sgd.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>

namespace minitensor::nn::test
{
namespace
{
class TwoLayerModel
{
  public:
    struct ParameterTree
    {
        Linear::ParameterTree hidden;
        Linear::ParameterTree output;

      private:
        template <typename Self, typename Visitor>
        static void visit_members(Self &self, Visitor &visitor)
        {
            visitor.child("hidden", self.hidden);
            visitor.child("output", self.output);
        }

        friend class ::minitensor::nn::ParameterTreeAccess;
    };

    ParameterTree initialize(ParameterBuilder &builder, RandomKey key, TensorOptions options = {}) const
    {
        const auto [hidden_key, output_key] = split(key);
        return {hidden_.initialize(builder, hidden_key, options), output_.initialize(builder, output_key, options)};
    }

    Tensor operator()(const ParameterMap &parameters, const ParameterTree &tree, const Tensor &input) const
    {
        return output_(parameters, tree.output, tanh(hidden_(parameters, tree.hidden, input)));
    }

  private:
    Linear hidden_{2, 3};
    Linear output_{3, 1, false};
};
} // namespace

TEST(ModuleTest, SharesOneBuilderAndMapAcrossChildren)
{
    const TwoLayerModel model;
    const auto snapshot = initialize(model, RandomKey{101});
    EXPECT_EQ(snapshot.parameters().size(), 3U);
    EXPECT_EQ(snapshot.value(snapshot.tree().hidden.weight()).shape(), (Shape{3, 2}));
    EXPECT_EQ(snapshot.value(snapshot.tree().output.weight()).shape(), (Shape{1, 3}));

    std::vector<std::string> paths;
    for_each_named_parameter(snapshot.tree(), [&](std::string_view path, ParameterId id) {
        paths.emplace_back(path);
        EXPECT_TRUE(snapshot.parameters().contains(id));
    });
    EXPECT_EQ(paths, (std::vector<std::string>{"hidden.weight", "hidden.bias", "output.weight"}));

    // Each child uses the same full map with only its own subtree.
    const Tensor input = from_data({1.0F, 2.0F}, Shape{1, 2});
    const Linear hidden{2, 3};
    const Linear output{3, 1, false};
    const Tensor expected = output(snapshot.parameters(), snapshot.tree().output,
                                   tanh(hidden(snapshot.parameters(), snapshot.tree().hidden, input)));
    EXPECT_EQ(to_vector(forward(model, snapshot, input)), to_vector(expected));
}

TEST(ModuleTest, PassesTensorOptionsToEveryChildWithoutEvaluating)
{
    const TwoLayerModel model;
    const auto snapshot = initialize(model, RandomKey{103}, TensorOptions{DType::Float32, Device::cpu(1)});
    for (const auto &[id, state] : snapshot.parameters())
    {
        (void)id;
        EXPECT_EQ(state.value.device(), Device::cpu(1));
    }
}

TEST(ModuleTest, TrainsThroughRepeatedSnapshotUpdates)
{
    const TwoLayerModel model;
    auto snapshot = initialize(model, RandomKey{107});
    const auto original = snapshot;
    const Tensor input = from_data({1.0F, 2.0F}, Shape{1, 2});
    const Tensor target = full(Shape{1, 1}, 1.0F);
    const auto loss_for = [&](const auto &parameters) { return mse_loss(forward(model, parameters, input), target); };
    const float initial_loss = item(loss_for(snapshot));
    const SGD optimizer{0.05F};
    auto state = optimizer.initialize(snapshot);

    for (std::size_t step = 0; step < 10; ++step)
    {
        const auto gradients = parameter_gradients(loss_for(snapshot), snapshot);
        ASSERT_EQ(gradients.size(), 3U);
        auto [updated, next_state] = optimizer.step(snapshot, state, gradients);
        snapshot = leafify_parameter_values(updated);
        state = leafify_optimizer_state(std::move(next_state));
    }

    EXPECT_LT(item(loss_for(snapshot)), initial_loss);
    EXPECT_FLOAT_EQ(item(loss_for(original)), initial_loss);
    EXPECT_EQ(snapshot.tree().hidden.weight(), original.tree().hidden.weight());
    EXPECT_EQ(snapshot.tree().output.weight(), original.tree().output.weight());
}
} // namespace minitensor::nn::test
