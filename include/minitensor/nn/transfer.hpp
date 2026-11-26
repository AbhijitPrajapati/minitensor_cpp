#pragma once

#include <minitensor/data.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
// Transfer each parameter value to the specified device.
template <typename Tree>
[[nodiscard]] ParameterSnapshot<Tree> to_device(ParameterSnapshot<Tree> snapshot, Device device)
{
    return std::move(snapshot).transform([device](ParameterId, const ParameterState &state) {
        return ParameterState{to_device(state.value, device), state.metadata};
    });
}
} // namespace minitensor::nn