#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/persistence/detail/adamw.hpp>
#include <minitensor/nn/persistence/detail/optimizer_persistence.hpp>
#include <minitensor/nn/persistence/detail/rmsprop.hpp>
#include <minitensor/nn/persistence/detail/sgd.hpp>
#include <minitensor/nn/persistence/detail/snapshot_binding.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn
{
constexpr persistence::detail::RecordHeaderSchema checkpoint_schema{"checkpoint", 1};

template <typename Tree, typename Optimizer>
struct Checkpoint
{
    ParameterSnapshot<Tree> snapshot;
    Optimizer optimizer;
    typename Optimizer::State optimizer_state;
};

// This small inner function allows snapshot-only capture to occur without an explicit void type argument.
namespace detail
{
template <typename Optimizer, typename Tree>
[[nodiscard]] persistence::Record capture_checkpoint_components(
    const ParameterSnapshot<Tree> &snapshot, typename OptimizerPersistence<Optimizer>::Capture &&capture)
{
    // Run the capture accumulator while generating the snapshot components.
    auto [paths, states] =
        capture_snapshot_components(snapshot, [&](std::size_t index, ParameterId id, const ParameterState &parameter) {
            capture.add_parameter(index, id, parameter);
        });

    auto record = persistence::detail::make_object_record(checkpoint_schema);
    record.add("paths", std::move(paths));
    record.add("states", std::move(states));

    // Add the optimizer payload if present.
    if constexpr (!std::is_void_v<Optimizer>)
    {
        record.add(OptimizerPersistence<Optimizer>::payload_name, std::move(capture).finish());
    }
    return record;
}
} // namespace detail

template <typename Tree>
[[nodiscard]] persistence::Record capture_checkpoint(const ParameterSnapshot<Tree> &snapshot)
{
    return detail::capture_checkpoint_components<void>(snapshot, detail::OptimizerPersistence<void>::Capture{});
}

template <typename Tree, typename Optimizer>
[[nodiscard]] persistence::Record capture_checkpoint(const ParameterSnapshot<Tree> &snapshot,
                                                     const Optimizer &optimizer, const typename Optimizer::State &state)
{
    return detail::capture_checkpoint_components<Optimizer>(
        snapshot, typename detail::OptimizerPersistence<Optimizer>::Capture{optimizer, state});
}

// Returns a checkpoint for optimizer restoration and a record for snapshot-only restoration.
template <typename Optimizer = void, typename Tree>
[[nodiscard]] auto restore_checkpoint(const persistence::Record &record, ParameterSnapshot<Tree> target)
{
    using Adapter = detail::OptimizerPersistence<Optimizer>;
    if constexpr (std::is_void_v<Optimizer>)
    {
        persistence::detail::validate_object_record(record, checkpoint_schema, {"paths", "states"});
    }
    else
    {
        persistence::detail::validate_object_record(record, checkpoint_schema,
                                                    {"paths", "states", Adapter::payload_name});
        Adapter::validate_payload(record.get(Adapter::payload_name).template get<persistence::Record>());
    }

    const auto &paths = record.get("paths").get<persistence::Record>();
    const auto &states = record.get("states").get<persistence::Value::List>();
    auto [snapshot, binding] = detail::restore_snapshot_components(paths, states, std::move(target));
    if constexpr (std::is_void_v<Optimizer>)
    {
        return std::move(snapshot);
    }
    else
    {
        const auto &payload = record.get(Adapter::payload_name).template get<persistence::Record>();
        auto optimizer = Adapter::restore_optimizer(payload);
        auto state = Adapter::restore_state(payload, binding, snapshot.parameters());
        return Checkpoint<Tree, Optimizer>{std::move(snapshot), std::move(optimizer), std::move(state)};
    }
}
} // namespace minitensor::nn
