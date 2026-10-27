#pragma once

#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn::detail
{
template <typename Function>
concept ParameterGradientTransform =
    std::invocable<Function &, ParameterId, const Tensor &, const Tensor &> &&
    std::convertible_to<std::invoke_result_t<Function &, ParameterId, const Tensor &, const Tensor &>, Tensor>;

// This function traverses and validates each unique parameter, (skipping frozen parameters)
// finds the corresponding gradient, and invokes the callback with the validated pair.
template <typename Tree, ParameterGradientTransform Update>
[[nodiscard]] ParameterSnapshot<Tree> transform_parameter_values_with_gradients(ParameterSnapshot<Tree> snapshot,
                                                                                const ParameterGradients &gradients,
                                                                                Update &&update)
{
    std::size_t update_count = 0;

    ParameterSnapshot<Tree> updated =
        transform_parameters(std::move(snapshot), [&](ParameterId id, const ParameterState &state) {
            if (!state.metadata.trainable)
            {
                return state;
            }
            if (!is_floating_point(state.value.dtype()))
            {
                throw std::invalid_argument{"Optimizer requires floating-point trainable parameters"};
            }

            // Find and validate gradient.
            const auto gradient_position = gradients.find(id);
            if (gradient_position == gradients.end())
            {
                throw std::invalid_argument{"parameter gradient not found"};
            }
            const Tensor &gradient = gradient_position->second;
            if (gradient.shape() != state.value.shape() || gradient.dtype() != state.value.dtype() ||
                gradient.device() != state.value.device())
            {
                throw std::invalid_argument{"parameter gradient does not match parameter spec"};
            }

            // Perform the update.
            Tensor updated_value = std::invoke(update, id, state.value, gradient);
            ++update_count;
            return ParameterState{std::move(updated_value), state.metadata};
        });

    // Ensure that gradients are exhaustive.
    if (update_count != gradients.size())
    {
        throw std::invalid_argument{"parameter gradients contain extra gradients"};
    }

    return updated;
}
} // namespace minitensor::nn::detail
