#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/autograd.hpp>
#include <minitensor/tensor.hpp>

#include "parameter.hpp"

namespace minitensor::nn
{

using ParameterGradients = std::unordered_map<ParameterId, Tensor>;

// Produces gradients for each unique parameter in the tree.
template <typename Tree>
[[nodiscard]] ParameterGradients parameter_gradients(const Tensor &loss, const ParameterSnapshot<Tree> &snapshot)
{
    // Collect parameter values
    std::vector<ParameterId> ids;
    std::vector<Tensor> values;

    for (const auto &[id, state] : snapshot.parameters())
    {
        if (state.metadata.trainable)
        {
            ids.push_back(id);
            values.push_back(state.value);
        }
    }

    std::vector<Tensor> gradients = grad(loss, values);

    // Build the output mapping
    ParameterGradients result;
    result.reserve(snapshot.parameters().size());

    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        result.emplace(ids[index], std::move(gradients[index]));
    }

    return result;
}
} // namespace minitensor::nn
