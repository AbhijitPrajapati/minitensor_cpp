#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn
{

constexpr persistence::detail::RecordHeaderSchema snapshot_schema{"snapshot", 1};

template <typename Tree>
[[nodiscard]] persistence::Record capture_snapshot(const ParameterSnapshot<Tree> &snapshot)
{
    const std::size_t size = snapshot.parameters().size();

    // Create a bidirectional mapping between parameter IDs and record indices.
    std::vector<ParameterId> parameter_ids;
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    parameter_ids.reserve(size);
    index_by_id.reserve(size);
    nn::for_each_parameter(snapshot.tree(), [&](ParameterId id) {
        // Assign indices in first-encounter tree order. Tied IDs are only counted once.
        if (index_by_id.try_emplace(id, parameter_ids.size()).second)
        {
            parameter_ids.push_back(id);
        }
    });

    persistence::Record record = persistence::detail::make_object_record(snapshot_schema);

    persistence::Record paths;
    persistence::Value::List states;

    states.reserve(size);
    for (ParameterId id : parameter_ids)
    {
        const ParameterState &parameter = snapshot.get(id);
        persistence::Record state;
        state.add("value", parameter.value);
        state.add("trainability", parameter.metadata.trainable);
        states.emplace_back(std::move(state));
    }

    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, ParameterId id) {
        paths.add(path, static_cast<std::uint64_t>(index_by_id.at(id)));
    });

    record.add("paths", std::move(paths));
    record.add("states", std::move(states));
    return record;
}

template <typename Tree>
[[nodiscard]] ParameterSnapshot<Tree> restore_snapshot(const persistence::Record &record,
                                                       ParameterSnapshot<Tree> target)
{
    persistence::detail::validate_object_record(record, snapshot_schema, {"paths", "states"});

    const auto &paths = record.get("paths").get<persistence::Record>();
    const auto &saved_states = record.get("states").get<persistence::Value::List>();
    std::unordered_map<ParameterId, std::size_t> target_to_state;
    target_to_state.reserve(target.parameters().size());
    std::vector<std::optional<ParameterId>> state_to_target(saved_states.size());

    std::size_t matched_paths = 0;
    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, ParameterId id) {
        if (!paths.contains(path))
        {
            throw std::invalid_argument{"parameter snapshot is missing a target path"};
        }
        const auto saved_index = paths.get(path).get<std::uint64_t>();
        if (saved_index >= saved_states.size())
        {
            throw std::out_of_range{"snapshot state index is out of range"};
        }
        const auto index = static_cast<std::size_t>(saved_index);
        const auto [position, inserted] = target_to_state.try_emplace(id, index);
        if (!inserted && position->second != index)
        {
            throw std::invalid_argument{"tied target parameter references distinct saved states"};
        }
        auto &target_id = state_to_target[index];
        if (target_id && *target_id != id)
        {
            throw std::invalid_argument{"tied saved parameter references distinct target parameters"};
        }
        target_id = id;
        ++matched_paths;
    });

    if (matched_paths != paths.size())
    {
        throw std::invalid_argument{"parameter snapshot contains unexpected paths"};
    }

    std::vector<ParameterState> states;
    states.reserve(saved_states.size());
    for (std::size_t index = 0; index < saved_states.size(); ++index)
    {
        const auto &id = state_to_target[index];
        if (!id)
        {
            throw std::invalid_argument{"snapshot contains unreferenced state"};
        }
        const auto &saved = saved_states[index].get<persistence::Record>();
        if (!saved.contains({"value", "trainability"}, true))
        {
            throw std::invalid_argument{"state record does not contain the expected fields"};
        }
        const auto &value = saved.get("value").get<Tensor>();
        const auto &expected = target.get(*id).value;
        if (value.shape() != expected.shape() || value.dtype() != expected.dtype() ||
            value.device() != expected.device())
        {
            throw std::invalid_argument{"saved parameter specification does not match target"};
        }
        states.push_back(ParameterState{value, {saved.get("trainability").get<bool>()}});
    }

    return std::move(target).transform(
        [&](ParameterId id, const ParameterState &) { return std::move(states[target_to_state.at(id)]); });
}
} // namespace minitensor::nn
