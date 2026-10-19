#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minitensor/data.hpp"
#include "minitensor/nn/gradients.hpp"
#include "minitensor/nn/leafification.hpp"
#include "minitensor/nn/loss.hpp"
#include "minitensor/nn/modules/linear.hpp"
#include "minitensor/nn/optimizers/adamw.hpp"
#include "minitensor/ops/creation.hpp"
#include "minitensor/ops/elementwise.hpp"
#include "minitensor/random.hpp"
#include "minitensor/tensor.hpp"
#include "minitensor/types.hpp"

namespace mt = minitensor;

struct SpiralDataset final
{
    mt::Tensor inputs;
    mt::Tensor classes;
};

[[nodiscard]] static SpiralDataset make_spiral_dataset(std::size_t points_per_class, mt::RandomKey noise_key)
{
    constexpr std::size_t num_classes = 3;

    if (points_per_class < 2)
    {
        throw std::invalid_argument{"use at least two points per class"};
    }

    const std::size_t sample_count = num_classes * points_per_class;

    const std::vector<float> noise =
        mt::to_vector(mt::normal(mt::Shape{static_cast<mt::Extent>(sample_count)}, 0.0F, 0.2F, noise_key));

    // The x and y coordinate are in subsequent pairs such that each row
    // in the resulting Tensor will correctly contain a coordinate pair.
    std::vector<float> input_data(sample_count * 2);

    std::vector<std::int64_t> class_data(sample_count);

    for (std::size_t class_index = 0; class_index < num_classes; ++class_index)
    {
        for (std::size_t point = 0; point < points_per_class; ++point)
        {
            const std::size_t sample_index = class_index * points_per_class + point;

            const float radius = static_cast<float>(point) / static_cast<float>(points_per_class);

            // Each spiral starts at a different angle depending on the class index,
            // and increases as the radius increases, with some added noise.
            const float angle = 2.094F * static_cast<float>(class_index) + 4.0F * radius + noise[sample_index];

            input_data[sample_index * 2] = radius * std::cos(angle);
            input_data[sample_index * 2 + 1] = radius * std::sin(angle);
            class_data[sample_index] = static_cast<std::int64_t>(class_index);
        }
    }

    return {
        mt::from_data(input_data, mt::Shape{static_cast<mt::Extent>(sample_count), 2}),
        mt::from_data(class_data, mt::Shape{static_cast<mt::Extent>(sample_count)}),
    };
}

// The model's parameter tree is defined seperately from its computation.
class ThreeSpiralModel final
{
  public:
    class Parameters final
    {
      public:
        using LinearParameters = mt::nn::Linear::Parameters;

        [[nodiscard]] const LinearParameters &hidden() const noexcept
        {
            return hidden_;
        }

        [[nodiscard]] const LinearParameters &output() const noexcept
        {
            return output_;
        }

      private:
        Parameters(LinearParameters hidden, LinearParameters output)
            : hidden_{std::move(hidden)}, output_{std::move(output)}
        {
        }

        // As part of the parameter traversal contract, this function allows
        // visitors to identify the children modules. Names such as "hidden.weight"
        // and "output.bias" will be automatically constructed.
        template <typename Self, typename Visitor>
        static void visit_members(Self &self, Visitor &visitor)
        {
            visitor.child("hidden", self.hidden_);
            visitor.child("output", self.output_);
        }

        // This class encapsulates the set of two linear parameters needed
        // for the model to run.
        LinearParameters hidden_;
        LinearParameters output_;

        friend class ThreeSpiralModel;

        // ParameterTreeAccess must be declared a friend in order to participate
        // in parameter traversal.
        friend class minitensor::nn::ParameterTreeAccess;
    };

    explicit ThreeSpiralModel(mt::Extent hidden_size, bool use_bias = true)
        : hidden_{2, hidden_size, use_bias}, output_{hidden_size, 3, use_bias}
    {
    }

    // The user retains and manages the full set of model parameters.
    [[nodiscard]] Parameters initialize(mt::RandomKey key, mt::TensorOptions options = {}) const
    {
        // Seperate key for each Linear module
        auto [hidden_key, output_key] = mt::split(key);

        return Parameters{
            hidden_.initialize(hidden_key, options),
            output_.initialize(output_key, options),
        };
    }

    [[nodiscard]] mt::Tensor operator()(const Parameters &parameters, const mt::Tensor &input) const
    {
        // TanH introduces non-linearity, which is essential for learning non-linear
        // relationships. Without it, the two linear transformations would compose to
        // form a single one.
        mt::Tensor hidden = mt::tanh(hidden_(parameters.hidden(), input));
        return output_(parameters.output(), hidden);
    }

  private:
    // The model instance only owns submodule instances, not their parameters.
    mt::nn::Linear hidden_;
    mt::nn::Linear output_;
};

int main()
{
    // Random keys are explicitly managed by the user.
    auto [dataset_key, model_key] = mt::split(mt::RandomKey{54});
    auto [train_key, test_key] = mt::split(dataset_key);

    SpiralDataset train_set = make_spiral_dataset(100, train_key);

    const mt::Tensor &train_inputs = train_set.inputs;
    const mt::Tensor &train_classes = train_set.classes;

    ThreeSpiralModel model(64);
    auto parameters = model.initialize(model_key);

    // Optimizer state is seperately managed from the optimizer instance.
    mt::nn::AdamW optimizer(0.01F, 0.0001F);
    auto optimizer_state = optimizer.initialize(parameters);

    for (std::size_t epoch = 1; epoch <= 500; ++epoch)
    {
        // The user provides the parameters along with the inputs during the forward pass.
        mt::Tensor logits = model(parameters, train_inputs);
        mt::Tensor loss = mt::nn::categorical_cross_entropy_with_logits(logits, train_classes);

        // No computation is performed until the loss is requested.
        std::cout << "Epoch: " << epoch << " Loss: " << mt::item<float>(loss) << "\n";

        // Gradient computation only adds to the lazy graph. It does not execute
        // until resulting Tensors are evaluated.
        auto gradients = mt::nn::parameter_gradients(loss, parameters);

        // Parameters and optimizer state are immutable by design.
        auto [new_parameters, new_state] = optimizer.step(std::move(parameters), std::move(optimizer_state), gradients);

        // In order to avoid an unmanagable computation graph, both states
        // must be leafified at the end of each training step. This process
        // inherently involve evaluating the Tensors.
        parameters = mt::nn::leafify_parameter_values(std::move(new_parameters));
        optimizer_state = mt::nn::leafify_optimizer_state(std::move(new_state));
    }

    // Testing to check for generalizability
    SpiralDataset test_set = make_spiral_dataset(20, test_key);

    const mt::Tensor &test_inputs = test_set.inputs;
    const mt::Tensor &test_classes = test_set.classes;

    mt::Tensor test_logits = model(parameters, test_inputs);
    mt::Tensor test_loss = mt::nn::categorical_cross_entropy_with_logits(test_logits, test_classes);

    std::cout << "\n" << "Test Loss: " << mt::item<float>(test_loss);
}
