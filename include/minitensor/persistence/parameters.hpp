#pragma once

#include <utility>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/persistence/record.hpp>

#include "detail/parameter_binding.hpp"
#include "minitensor/persistence/detail/parameters.hpp"

namespace minitensor::persistence
{

// Captures parameter snapshot data into a record without evaluating tensors.
template <typename Tree>
[[nodiscard]] Record capture_parameters(const nn::ParameterSnapshot<Tree> &snapshot)
{
    const detail::ParameterBinding binding = detail::capture_binding(snapshot);
    return detail::capture_snapshot(snapshot, binding);
}

template <typename Tree>
[[nodiscard]] nn::ParameterSnapshot<Tree> restore_parameters(const Record &record, nn::ParameterSnapshot<Tree> target)
{
    // Parse snapshot record.
    detail::ParsedParameters saved = detail::parse_parameters(record);
    // Recover bindings.
    const detail::ParameterBinding binding = detail::restore_binding(saved.references, target);
    return detail::restore_snapshot(std::move(target), binding, saved.states);
}
} // namespace minitensor::persistence
