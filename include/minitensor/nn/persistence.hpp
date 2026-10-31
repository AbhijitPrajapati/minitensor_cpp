#pragma once

#include "minitensor/nn/parameter_id.hpp"
#include "minitensor/nn/parameter_tree.hpp"
#include "minitensor/tensor.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <initializer_list>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/persistence/record.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minitensor::nn
{

// Captures snapshot data into a record. Does not evaluate Tensors.
template <typename Tree>
persistence::Record capture_parameters(const ParameterSnapshot<Tree> &snapshot)
{
    persistence::Record record;
    record.add("kind", "parameter_snapshot");
    record.add("version", uint64_t{1});

    // Each tree path is associated with a record-specific state index
    // (separate from the process ParameterId).
    persistence::Record paths;
    // Tensor values and parameter metadata are preserved in state index order.
    persistence::Value::List states;

    std::unordered_map<ParameterId, std::uint64_t> state_indices;

    const std::size_t size = snapshot.parameters().size();
    states.reserve(size);
    state_indices.reserve(size);

    for_each_named_parameter(snapshot.tree(), [&](std::string_view path, const ParameterId parameter_id) {
        // Use the current size of the states list to determine the next ID.
        const auto [position, inserted] =
            state_indices.try_emplace(parameter_id, static_cast<std::uint64_t>(states.size()));
        if (inserted)
        {
            // Add a new state entry if the parameter is unseen.
            const ParameterState &parameter = snapshot.get(parameter_id);
            persistence::Record state;
            state.add("value", parameter.value);
            state.add("trainability", parameter.metadata.trainable);
            states.emplace_back(std::move(state));
        }
        // Record the path along with its record index regardless to preserve weight ties.
        paths.add(path, position->second);
    });

    record.add("paths", std::move(paths));
    record.add("states", std::move(states));
    return record;
}

template <typename Tree>
ParameterSnapshot<Tree> restore_parameters(const persistence::Record &record, ParameterSnapshot<Tree> target)
{
    // Confirm that required fields are present.
    const auto require_fields = [](const persistence::Record &obj, std::initializer_list<std::string_view> fields) {
        if (obj.size() != fields.size())
        {
            throw std::invalid_argument{"snapshot record has unexpected fields"};
        }
        for (std::string_view field : fields)
        {
            if (!obj.contains(field))
            {
                throw std::invalid_argument{"snapshot record is missing a required field"};
            }
        }
    };

    require_fields(record, {"kind", "version", "paths", "states"});

    // Confirm parameter snapshot kind and version.
    const auto &kind = record.get("kind").get<std::string>();
    if (kind != "parameter_snapshot")
    {
        throw std::invalid_argument{"record is not a parameter snapshot"};
    }
    const auto version = record.get("version").get<std::uint64_t>();
    if (version != 1)
    {
        throw std::invalid_argument{"unsupported parameter snapshot version"};
    }

    // Parse state records upfront into ParameterState
    const auto &saved_states = record.get("states").get<persistence::Value::List>();
    std::vector<ParameterState> states;
    states.reserve(saved_states.size());

    for (const persistence::Value &entry : saved_states)
    {
        const auto &state_record = entry.get<persistence::Record>();
        require_fields(state_record, {"value", "trainability"});
        const auto &value = state_record.get("value").get<Tensor>();
        const bool trainability = state_record.get("trainability").get<bool>();
        states.push_back(ParameterState{value, ParameterMetadata{trainability}});
    }

    // Validate every path reference, ensuring that each state is referenced at least once.
    const auto &paths = record.get("paths").get<persistence::Record>();
    std::vector<bool> referenced(states.size(), false);

    for (const auto &[path, entry] : paths)
    {
        const auto index = entry.get<std::uint64_t>();
        if (index >= states.size())
        {
            throw std::out_of_range{"snapshot state index is out of range"};
        }
        referenced[static_cast<std::size_t>(index)] = true;
    }

    if (std::find(referenced.begin(), referenced.end(), false) != referenced.end())
    {
        throw std::invalid_argument{"snapshot contains unreferenced state"};
    }

    // Parameter ID -> record state
    std::unordered_map<ParameterId, std::size_t> target_to_state;
    target_to_state.reserve(target.parameters().size());

    // Record state -> parameter ID
    std::vector<std::optional<ParameterId>> state_to_target(states.size());

    // Both mappings are needed to enforce strict snapshot adherance for weight tying. Only target_to_state is used for
    // the transformation.

    std::size_t matched_paths = 0;
    for_each_named_parameter(target.tree(), [&](std::string_view path, ParameterId id) {
        if (!paths.contains(path))
        {
            throw std::invalid_argument{"parameter snapshot is missing a target path"};
        }

        const auto index = static_cast<std::size_t>(paths.get(path).get<std::uint64_t>());
        const auto [position, inserted] = target_to_state.try_emplace(id, index);

        // The record has assigned different state indices to tied snapshot parameters.
        if (!inserted && position->second != index)
        {
            throw std::invalid_argument{"tied target parameter references distinct saved states"};
        }

        auto &target_id = state_to_target[index];

        // The snapshot contains tied parameters that correspond to distinct record state indices.
        if (target_id && *target_id != id)
        {
            throw std::invalid_argument{"tied saved parameter references distinct target parameters"};
        }

        target_id = id;
        ++matched_paths;
    });

    // Snapshot and target paths must be the same.
    if (matched_paths != paths.size())
    {
        throw std::invalid_argument{"parameter snapshot contains unexpected paths"};
    }

    // Ensure that tensor specifications match.
    for (const auto &[id, target_state] : target.parameters())
    {
        const std::size_t index = target_to_state.at(id);
        const Tensor &saved_value = states[index].value;
        const Tensor &target_value = target_state.value;

        if (saved_value.shape() != target_value.shape() || saved_value.dtype() != target_value.dtype() ||
            saved_value.device() != target_value.device())
        {
            throw std::invalid_argument{"saved parameter specification does not match target"};
        }
    }

    // Transform the snapshot
    return std::move(target).transform(
        [&](ParameterId parameter_id, const ParameterState &) { return states[target_to_state.at(parameter_id)]; });
}
} // namespace minitensor::nn