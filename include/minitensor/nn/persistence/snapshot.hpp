#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/persistence/detail/parameter_binding.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn
{

constexpr persistence::detail::RecordHeaderSchema snapshot_schema{"snapshot", 1};

template <typename Tree>
[[nodiscard]] persistence::Record capture_snapshot(const ParameterSnapshot<Tree> &snapshot)
{
    persistence::Value::List states;
    states.reserve(snapshot.parameters().size());

    persistence::Record paths =
        detail::capture_parameter_paths(snapshot, [&](std::size_t, ParameterId, const ParameterState &parameter) {
            persistence::Record state;
            state.add("value", parameter.value);
            state.add("trainability", parameter.metadata.trainable);
            states.emplace_back(std::move(state));
        });

    persistence::Record record = persistence::detail::make_object_record(snapshot_schema);

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
    const auto binding = detail::bind_parameter_paths(paths, target, saved_states.size());

    std::vector<ParameterState> states;
    states.reserve(saved_states.size());
    for (std::size_t index = 0; index < saved_states.size(); ++index)
    {
        const ParameterId id = *binding.id_by_index[index];
        const auto &saved = saved_states[index].get<persistence::Record>();
        if (!saved.contains({"value", "trainability"}, true))
        {
            throw std::invalid_argument{"state record does not contain the expected fields"};
        }
        const auto &value = saved.get("value").get<Tensor>();
        const auto &expected = target.get(id).value;
        if (value.shape() != expected.shape() || value.dtype() != expected.dtype() ||
            value.device() != expected.device())
        {
            throw std::invalid_argument{"saved parameter specification does not match target"};
        }
        states.push_back(ParameterState{value, {saved.get("trainability").get<bool>()}});
    }

    return std::move(target).transform(
        [&](ParameterId id, const ParameterState &) { return std::move(states[binding.index_by_id.at(id)]); });
}
} // namespace minitensor::nn
