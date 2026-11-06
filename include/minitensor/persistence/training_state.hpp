#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/persistence/record.hpp>

#include "detail/parameter_binding.hpp"
#include "minitensor/persistence/detail/optimizer_persistence.hpp"
#include "minitensor/persistence/detail/optimizers/adamw.hpp"
#include "minitensor/persistence/detail/parameters.hpp"

namespace minitensor::persistence
{

template <typename Optimizer, typename Tree>
[[nodiscard]] Record capture_training_state(const nn::ParameterSnapshot<Tree> &snapshot,
                                            const typename Optimizer::State &optimizer_state)
{
    using Adapter = detail::OptimizerPersistence<Optimizer>;

    Record record;
    record.add("kind", "training_state");
    record.add("version", std::uint64_t{2});

    // Create shared binding.
    const detail::ParameterBinding binding = detail::capture_binding(snapshot);

    record.add("parameters", detail::capture_snapshot(snapshot, binding));
    record.add("optimizer_state", Adapter::capture_state(optimizer_state, snapshot.parameters(), binding));
    return record;
}

template <typename Tree, typename Optimizer>
struct TrainingState
{
    nn::ParameterSnapshot<Tree> snapshot;
    typename Optimizer::State optimizer_state;
};

template <typename Optimizer, typename Tree>
[[nodiscard]] TrainingState<Tree, Optimizer> restore_training_state(const Record &record,
                                                                    nn::ParameterSnapshot<Tree> target)
{
    using Adapter = detail::OptimizerPersistence<Optimizer>;

    // Validate header fields.
    if (!record.contains({"kind", "version", "parameters", "optimizer_state"}, true))
    {
        throw std::invalid_argument{"training state record does not contain required fields"};
    }

    // Confirm kind and version.
    const auto &kind = record.get("kind").get<std::string>();
    if (kind != "training_state")
    {
        throw std::invalid_argument{"record is not a training state"};
    }
    const auto version = record.get("version").get<std::uint64_t>();
    if (version != 2)
    {
        throw std::invalid_argument{"unsupported training state version"};
    }

    const auto &parameter_record = record.get("parameters").get<Record>();
    const auto &optimizer_state_record = record.get("optimizer_state").get<Record>();

    // Parse and bind parameters.
    detail::ParsedParameters saved = detail::parse_parameters(parameter_record);
    const detail::ParameterBinding binding = detail::restore_binding(saved.references, target);

    // Restore parameters and optimizer state.
    auto parameters = detail::restore_snapshot(std::move(target), binding, saved.states);
    auto optimizer_state =
        Adapter::restore_state(optimizer_state_record, parameters.parameters(), binding.parameter_ids);
    return {std::move(parameters), std::move(optimizer_state)};
}
} // namespace minitensor::persistence
