#pragma once

#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
// Parameter tensors must be leafified after each training step in order to
// release graphs from the previous step.

template <typename Tree>
[[nodiscard]] ParameterSnapshot<Tree> leafify_parameter_values(const ParameterSnapshot<Tree> &snapshot)
{
    // Collect IDs and values
    std::vector<ParameterId> ids;
    std::vector<Tensor> values;
    ids.reserve(snapshot.parameters().size());
    values.reserve(snapshot.parameters().size());

    for (const auto &[id, state] : snapshot.parameters())
    {
        ids.push_back(id);
        values.push_back(state.value);
    }

    std::vector<Tensor> leafified = leafify(values);

    ParameterMap::Map states;
    states.reserve(ids.size());

    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        states.emplace(ids[i], ParameterState{std::move(leafified[i]), snapshot.get(ids[i]).metadata});
    }

    return ParameterSnapshot<Tree>{snapshot.tree(), ParameterMap{std::move(states)}};
}
} // namespace minitensor::nn
