#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/nn/optimizers/sgd.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/persistence/record.hpp>

#include "minitensor/nn/persistence/detail/adamw_state.hpp"
#include "minitensor/nn/persistence/detail/snapshot_binding.hpp"

namespace minitensor::nn
{
constexpr persistence::detail::RecordHeaderSchema checkpoint_schema{"checkpoint", 1};

template <typename Tree>
[[nodiscard]] persistence::Record capture_checkpoint(const ParameterSnapshot<Tree> &snapshot, const AdamW &optimizer,
                                                     const AdamW::State &optimizer_state)
{
    detail::AdamWStateCodec state_capture{optimizer_state};
    auto [paths, states] = detail::capture_snapshot_components(
        snapshot, [&](std::size_t index, ParameterId id, const ParameterState &parameter) {
            state_capture.add_parameter(index, id, parameter);
        });

    persistence::Record adamw_record;
    adamw_record.add("learning_rate", optimizer.learning_rate());
    adamw_record.add("weight_decay", optimizer.weight_decay());
    adamw_record.add("beta1", optimizer.beta1());
    adamw_record.add("beta2", optimizer.beta2());
    adamw_record.add("epsilon", optimizer.epsilon());
    std::move(state_capture).finish(adamw_record);

    persistence::Record record = persistence::detail::make_object_record(checkpoint_schema);
    record.add("paths", std::move(paths));
    record.add("states", std::move(states));
    record.add("adamw", std::move(adamw_record));

    return record;
}

template <typename Tree, typename Optimizer>
struct Checkpoint
{
    ParameterSnapshot<Tree> snapshot;
    Optimizer optimizer;
    typename Optimizer::State optimizer_state;
};

template <typename Tree>
[[nodiscard]] Checkpoint<Tree, AdamW> restore_adamw_checkpoint(const persistence::Record &record,
                                                               ParameterSnapshot<Tree> target)
{
    persistence::detail::validate_object_record(record, checkpoint_schema, {"paths", "states", "adamw"});

    const auto &adamw = record.get("adamw").get<persistence::Record>();
    if (!adamw.contains({"learning_rate", "weight_decay", "beta1", "beta2", "epsilon", "step", "moments"}, true))
    {
        throw std::invalid_argument{"AdamW checkpoint does not contain the expected fields"};
    }
    AdamW optimizer{adamw.get("learning_rate").get<float>(), adamw.get("weight_decay").get<float>(),
                    adamw.get("beta1").get<float>(), adamw.get("beta2").get<float>(),
                    adamw.get("epsilon").get<float>()};

    const auto &paths = record.get("paths").get<persistence::Record>();
    const auto &saved_states = record.get("states").get<persistence::Value::List>();
    auto [snapshot, binding] = detail::restore_snapshot_components(paths, saved_states, std::move(target));
    auto state = detail::AdamWStateCodec::restore(adamw, binding, snapshot.parameters());
    return {std::move(snapshot), std::move(optimizer), std::move(state)};
}

template <typename Tree>
[[nodiscard]] persistence::Record capture_checkpoint(const ParameterSnapshot<Tree> &snapshot, const SGD &optimizer,
                                                     const SGD::State &)
{
    auto [paths, states] =
        detail::capture_snapshot_components(snapshot, [](std::size_t, ParameterId, const ParameterState &) {});
    persistence::Record sgd;
    sgd.add("learning_rate", optimizer.learning_rate());

    persistence::Record record = persistence::detail::make_object_record(checkpoint_schema);
    record.add("paths", std::move(paths));
    record.add("states", std::move(states));
    record.add("sgd", std::move(sgd));
    return record;
}

template <typename Tree>
[[nodiscard]] Checkpoint<Tree, SGD> restore_sgd_checkpoint(const persistence::Record &record,
                                                           ParameterSnapshot<Tree> target)
{
    persistence::detail::validate_object_record(record, checkpoint_schema, {"paths", "states", "sgd"});

    const auto &sgd = record.get("sgd").get<persistence::Record>();
    if (!sgd.contains({"learning_rate"}, true))
    {
        throw std::invalid_argument{"SGD checkpoint does not contain the expected fields"};
    }
    SGD optimizer{sgd.get("learning_rate").get<float>()};

    const auto &paths = record.get("paths").get<persistence::Record>();
    const auto &saved_states = record.get("states").get<persistence::Value::List>();
    return {
        detail::restore_snapshot_components(paths, saved_states, std::move(target)).first, std::move(optimizer), {}};
}
} // namespace minitensor::nn
