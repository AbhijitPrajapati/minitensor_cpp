#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_id.hpp>
#include <minitensor/nn/parameter_tree.hpp>

namespace minitensor::persistence::detail
{

// Represents a bidirectional mapping between parameter IDs and state indices.
struct ParameterBinding
{
    std::vector<nn::ParameterId> parameter_ids;
    std::unordered_map<nn::ParameterId, std::size_t> index_by_id;
};

// Creates a ParameterBinding using parameter IDs from the capture process.
template <typename Tree>
[[nodiscard]] ParameterBinding capture_binding(const nn::ParameterSnapshot<Tree> &snapshot)
{
    ParameterBinding output;
    const std::size_t size = snapshot.parameters().size();
    output.parameter_ids.reserve(size);
    output.index_by_id.reserve(size);
    nn::for_each_parameter(snapshot.tree(), [&](const nn::ParameterId parameter_id) {
        // Assign indices in first-encounter tree order. Tied IDs are only counted once.
        if (output.index_by_id.try_emplace(parameter_id, output.parameter_ids.size()).second)
        {
            output.parameter_ids.emplace_back(parameter_id);
        }
    });
    return output;
}

// A subcomponent of the parsed record containing the information required to restore a ParameterBinding.
struct ParameterReferences
{
    std::map<std::string, std::size_t, std::less<>> index_by_path;
    std::size_t state_count;
};

// Recovers a ParameterBinding using parameter IDs from the restoration process and the ParameterReferences.
template <typename Tree>
[[nodiscard]] ParameterBinding restore_binding(const ParameterReferences &saved,
                                               const nn::ParameterSnapshot<Tree> &target)
{
    // Parameter ID -> record state
    std::unordered_map<nn::ParameterId, std::size_t> target_to_state;
    target_to_state.reserve(target.parameters().size());

    // Record state -> parameter ID
    std::vector<std::optional<nn::ParameterId>> state_to_target(saved.state_count);

    // Athough only the latter mapping is returned, both mappings are needed to enforce strict snapshot adherance for
    // weight tying.

    std::size_t matched_paths = 0;
    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, nn::ParameterId id) {
        const auto saved_path = saved.index_by_path.find(path);
        if (saved_path == saved.index_by_path.end())
        {
            throw std::invalid_argument{"parameter snapshot is missing a target path"};
        }

        const std::size_t index = saved_path->second;
        if (index >= saved.state_count)
        {
            throw std::out_of_range{"saved parameter index is out of range"};
        }

        const auto [position, inserted] = target_to_state.try_emplace(id, index);

        // A tied target ID must correspond to one saved state.
        if (!inserted && position->second != index)
        {
            throw std::invalid_argument{"tied target parameter references distinct saved states"};
        }

        auto &target_id = state_to_target[index];

        // A tied saved state must correspond to one target ID.
        if (target_id && *target_id != id)
        {
            throw std::invalid_argument{"tied saved parameter references distinct target parameters"};
        }

        target_id = id;
        ++matched_paths;
    });

    // Snapshot and target path count must be the same.
    if (matched_paths != saved.index_by_path.size())
    {
        throw std::invalid_argument{"parameter snapshot contains unexpected paths"};
    }

    // Convert state-to-target mapping into dense vector.
    std::vector<nn::ParameterId> parameter_ids;
    parameter_ids.reserve(state_to_target.size());
    for (const auto &id : state_to_target)
    {
        // std::optional was used to sidestep the lack of default construction. Every ID should be populated.
        if (!id)
        {
            throw std::invalid_argument{"saved parameter state has no target"};
        }
        parameter_ids.push_back(id.value());
    }
    return {std::move(parameter_ids), std::move(target_to_state)};
}

} // namespace minitensor::persistence::detail
