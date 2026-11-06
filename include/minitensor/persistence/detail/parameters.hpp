#pragma once

#include <minitensor/nn/parameter.hpp>
#include <minitensor/persistence/record.hpp>
#include <span>

#include "minitensor/persistence/detail/parameter_binding.hpp"

namespace minitensor::persistence::detail
{
struct ParsedParameters
{
    ParameterReferences references;
    std::vector<nn::ParameterState> states;
};

[[nodiscard]] ParsedParameters parse_parameters(const Record &record);

template <typename Tree>
[[nodiscard]] Record capture_snapshot(const nn::ParameterSnapshot<Tree> &snapshot, const ParameterBinding &binding)
{
    Record record;
    record.add("kind", "parameter_snapshot");
    record.add("version", std::uint64_t{1});

    Record paths;
    Value::List states;

    // Emit each unique parameter state in the order assigned by the index.
    states.reserve(binding.parameter_ids.size());
    for (const nn::ParameterId parameter_id : binding.parameter_ids)
    {
        const nn::ParameterState &parameter = snapshot.get(parameter_id);
        Record state;
        state.add("value", parameter.value);
        state.add("trainability", parameter.metadata.trainable);
        states.emplace_back(std::move(state));
    }

    // Every path is retained regardless of tying.
    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, const nn::ParameterId parameter_id) {
        paths.add(path,
                  static_cast<std::uint64_t>(binding.index_by_id.at(parameter_id))); // Retrieve the correct index.
    });

    record.add("paths", std::move(paths));
    record.add("states", std::move(states));

    return record;
}

template <typename Tree>
[[nodiscard]] nn::ParameterSnapshot<Tree> restore_snapshot(nn::ParameterSnapshot<Tree> target,
                                                           const ParameterBinding &binding,
                                                           std::span<nn::ParameterState> states)
{
    // Validate Tensor specifications of saved states against target snapshot.
    for (std::size_t index = 0; index < states.size(); ++index)
    {
        const Tensor &saved = states[index].value;
        const Tensor &expected = target.get(binding.parameter_ids[index]).value;

        if (saved.shape() != expected.shape() || saved.dtype() != expected.dtype() ||
            saved.device() != expected.device())
        {
            throw std::invalid_argument{"saved parameter specification does not match target"};
        }
    }

    // Transform the target snapshot.
    return std::move(target).transform(
        [&](nn::ParameterId id, const nn::ParameterState &) { return std::move(states[binding.index_by_id.at(id)]); });
}

} // namespace minitensor::persistence::detail