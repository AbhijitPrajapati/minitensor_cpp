#pragma once

#include <cstddef>
#include <utility>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/persistence/record.hpp>

#include "minitensor/nn/persistence/detail/snapshot_binding.hpp"

namespace minitensor::nn
{

constexpr persistence::detail::RecordHeaderSchema snapshot_schema{"snapshot", 1};

template <typename Tree>
[[nodiscard]] persistence::Record capture_snapshot(const ParameterSnapshot<Tree> &snapshot)
{
    auto [paths, states] =
        detail::capture_snapshot_components(snapshot, [](std::size_t, ParameterId, const ParameterState &) {});
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

    return detail::restore_snapshot_components(paths, saved_states, std::move(target)).first;
}
} // namespace minitensor::nn
