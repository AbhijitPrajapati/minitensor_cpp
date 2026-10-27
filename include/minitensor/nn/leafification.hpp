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
[[nodiscard]] ParameterSnapshot<Tree> leafify_parameter_values(ParameterSnapshot<Tree> snapshot)
{
    // Collect values in the map's iteration order.
    std::vector<Tensor> values;
    values.reserve(snapshot.parameters().size());

    for (const auto &entry : snapshot.parameters())
    {
        values.push_back(entry.second.value);
    }

    std::vector<Tensor> leafified = leafify(values);

    // Perform the snapshot transformation.
    std::size_t index = 0;
    return std::move(snapshot).transform([&](ParameterId, const ParameterState &state) {
        return ParameterState{std::move(leafified[index++]), state.metadata};
    });
}
} // namespace minitensor::nn
