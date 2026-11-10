#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
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

namespace minitensor::nn::detail
{

// Assign one index per unique parameter, even when the codec emits no payload for it. Returns paths and states.
template <typename Tree, typename Function>
[[nodiscard]] std::pair<persistence::Record, persistence::Value::List> capture_snapshot_components(
    const ParameterSnapshot<Tree> &snapshot, Function &&on_parameter)
{
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    index_by_id.reserve(snapshot.parameters().size());

    persistence::Value::List states;
    states.reserve(snapshot.parameters().size());

    persistence::Record paths;
    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, ParameterId id) {
        const auto [position, inserted] = index_by_id.try_emplace(id, index_by_id.size());
        if (inserted)
        {
            const ParameterState &parameter = snapshot.get(id);

            persistence::Record state;
            state.add("value", parameter.value);
            state.add("trainability", parameter.metadata.trainable);
            states.emplace_back(std::move(state));

            std::invoke(on_parameter, position->second, id, parameter);
        }
        paths.add(path, static_cast<std::uint64_t>(position->second));
    });
    return {std::move(paths), std::move(states)};
}

struct ParameterBinding
{
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    std::vector<std::optional<ParameterId>> id_by_index;
};

template <typename Tree>
[[nodiscard]] std::pair<ParameterSnapshot<Tree>, ParameterBinding> restore_snapshot_components(
    const persistence::Record &paths, const persistence::Value::List &states, ParameterSnapshot<Tree> &&target)
{
    ParameterBinding binding;
    binding.index_by_id.reserve(target.parameters().size());
    binding.id_by_index.resize(states.size());

    std::size_t matched_paths = 0;
    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, ParameterId id) {
        if (!paths.contains(path))
        {
            throw std::invalid_argument{"saved parameter layout is missing a target path"};
        }
        const auto saved_index = paths.get(path).get<std::uint64_t>();
        if (saved_index >= states.size())
        {
            throw std::out_of_range{"saved parameter path index is out of range"};
        }
        const auto index = static_cast<std::size_t>(saved_index);
        const auto [position, inserted] = binding.index_by_id.try_emplace(id, index);
        if (!inserted && position->second != index)
        {
            throw std::invalid_argument{"tied target parameter references distinct saved indices"};
        }
        auto &target_id = binding.id_by_index[index];
        if (target_id && *target_id != id)
        {
            throw std::invalid_argument{"tied saved parameter references distinct target parameters"};
        }
        target_id = id;
        ++matched_paths;
    });

    if (matched_paths != paths.size())
    {
        throw std::invalid_argument{"saved parameter layout contains unexpected paths"};
    }

    // The checks above establish a one-to-one mapping. Equal counts ensure that every slot is bound.
    if (binding.index_by_id.size() != states.size())
    {
        throw std::invalid_argument{"saved parameter index has no target"};
    }

    std::vector<ParameterState> new_states;
    new_states.reserve(states.size());

    for (std::size_t index = 0; index < states.size(); ++index)
    {
        const ParameterId id = *binding.id_by_index[index];
        const auto &saved = states[index].get<persistence::Record>();
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
        new_states.push_back(ParameterState{value, {saved.get("trainability").get<bool>()}});
    }

    return {std::move(target).transform([&](ParameterId id, const ParameterState &) {
                return std::move(new_states[binding.index_by_id.at(id)]);
            }),
            std::move(binding)};
}

} // namespace minitensor::nn::detail
