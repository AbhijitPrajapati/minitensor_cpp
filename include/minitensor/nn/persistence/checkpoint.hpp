#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/nn/persistence/detail/optimizer_adapters/adamw.hpp>
#include <minitensor/nn/persistence/detail/optimizer_adapters/rmsprop.hpp>
#include <minitensor/nn/persistence/detail/optimizer_adapters/sgd.hpp>
#include <minitensor/nn/persistence/detail/optimizer_persistence.hpp>
#include <minitensor/nn/persistence/detail/parameter_binding.hpp>
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

namespace detail
{
// Captures optimizer data alongside a parameter snapshot using the given capture accumulator.
template <typename Optimizer, typename Tree>
[[nodiscard]] persistence::Record capture_checkpoint_components(
    const ParameterSnapshot<Tree> &snapshot, typename OptimizerPersistence<Optimizer>::Capture &&capture)
{
    // ID-index map used to account for tied parameters.
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    index_by_id.reserve(snapshot.parameters().size());

    persistence::Value::List states; // Individual state records
    states.reserve(snapshot.parameters().size());

    persistence::Record paths; // Exhaustive path index
    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, ParameterId id) {
        const auto [position, inserted] = index_by_id.try_emplace(id, index_by_id.size());
        if (inserted)
        {
            // Add a state record if the parameter is unseen.
            const ParameterState &parameter = snapshot.get(id);
            persistence::Record state;
            state.add("value", parameter.value);
            state.add("trainability", parameter.metadata.trainable);
            states.emplace_back(std::move(state));
            // Run the optimizer capture accumulator.
            capture.add_parameter(position->second, id, parameter);
        }
        // Record the path-index mapping.
        paths.add(path, static_cast<std::uint64_t>(position->second));
    });

    // Assemble the output record.
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
    // Delegate with a void optimizer type for snapshot-only capture.
    return detail::capture_checkpoint_components<void>(snapshot, detail::OptimizerPersistence<void>::Capture{});
}

template <typename Tree, typename Optimizer>
[[nodiscard]] persistence::Record capture_checkpoint(const ParameterSnapshot<Tree> &snapshot,
                                                     const Optimizer &optimizer, const typename Optimizer::State &state)
{
    // Delegate with the supplied optimizer type.
    return detail::capture_checkpoint_components<Optimizer>(
        snapshot, typename detail::OptimizerPersistence<Optimizer>::Capture{optimizer, state});
}

// Returns a Checkpoint with an optimizer, or a ParameterSnapshot without one.
template <typename Optimizer = void, typename Tree>
[[nodiscard]] auto restore_checkpoint(const persistence::Record &record, ParameterSnapshot<Tree> target)
{
    using Adapter = detail::OptimizerPersistence<Optimizer>;

    // Validate the record.
    if constexpr (std::is_void_v<Optimizer>)
    {
        persistence::detail::validate_object_record(record, checkpoint_schema, {"paths", "states"});
    }
    else
    {
        // Include the optimizer payload name.
        persistence::detail::validate_object_record(record, checkpoint_schema,
                                                    {"paths", "states", Adapter::payload_name});
        // Validate the optimizer payload.
        Adapter::validate_payload(record.get(Adapter::payload_name).template get<persistence::Record>());
    }

    // Restore the parameter snapshot and its coresponding parameter binding.
    const auto &paths = record.get("paths").get<persistence::Record>();
    const auto &states = record.get("states").get<persistence::Value::List>();

    // Construct and populate the binding.
    detail::ParameterBinding binding;
    binding.index_by_id.reserve(target.parameters().size());
    binding.id_by_index.resize(states.size());

    std::size_t matched_paths = 0;
    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, ParameterId id) {
        if (!paths.contains(path))
        {
            throw std::invalid_argument{"saved parameter layout is missing a target path"};
        }

        // Validate the saved index.
        const auto saved_index = paths.get(path).get<std::uint64_t>();
        if (saved_index >= states.size())
        {
            throw std::out_of_range{"saved parameter path index is out of range"};
        }
        const auto index = static_cast<std::size_t>(saved_index);

        // Populate the binding.

        const auto [position, inserted] = binding.index_by_id.try_emplace(id, index);
        // Ensure matching indices correspond to matching IDs.
        if (!inserted && position->second != index)
        {
            throw std::invalid_argument{"tied target parameter references distinct saved indices"};
        }
        // Ensure matching IDs correspond to matching indices.
        auto &target_id = binding.id_by_index[index];
        if (target_id && *target_id != id)
        {
            throw std::invalid_argument{"tied saved parameter references distinct target parameters"};
        }
        target_id = id;
        ++matched_paths;
    });

    // Ensure that the saved record does not contain extra paths.
    if (matched_paths != paths.size())
    {
        throw std::invalid_argument{"saved parameter layout contains unexpected paths"};
    }

    // Ensure that every parameter state slot is bound.
    if (binding.index_by_id.size() != states.size())
    {
        throw std::invalid_argument{"saved parameter index has no target"};
    }

    // Use the binding to construct parameter states in saved-index order.
    std::vector<ParameterState> new_states;
    new_states.reserve(states.size());

    for (std::size_t index = 0; index < states.size(); ++index)
    {
        // Retrieve and validate the state record.
        const auto &saved = states[index].get<persistence::Record>();
        if (!saved.contains({"value", "trainability"}, true))
        {
            throw std::invalid_argument{"state record does not contain the expected fields"};
        }
        const auto &value = saved.get("value").get<Tensor>();

        // Ensure that the record matches the parameter specification.
        const ParameterId id = *binding.id_by_index[index];
        const auto &expected = target.get(id).value;
        if (value.shape() != expected.shape() || value.dtype() != expected.dtype() ||
            value.device() != expected.device())
        {
            throw std::invalid_argument{"saved parameter specification does not match target"};
        }
        new_states.push_back(ParameterState{value, {saved.get("trainability").get<bool>()}});
    }

    // Transform the target snapshot, retrieving the saved index through the binding.
    auto snapshot = std::move(target).transform(
        [&](ParameterId id, const ParameterState &) { return std::move(new_states[binding.index_by_id.at(id)]); });

    if constexpr (std::is_void_v<Optimizer>)
    {
        // Return the snapshot if no optimizer is present.
        return std::move(snapshot);
    }
    else
    {
        const auto &payload = record.get(Adapter::payload_name).template get<persistence::Record>();
        // Restore optimizer configurations.
        auto optimizer = Adapter::restore_optimizer(payload);
        // Restore optimizer state using the binding.
        auto state = Adapter::restore_state(payload, binding, snapshot.parameters());
        return Checkpoint<Tree, Optimizer>{std::move(snapshot), std::move(optimizer), std::move(state)};
    }
}
} // namespace minitensor::nn
