#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn
{

constexpr persistence::detail::RecordHeaderSchema adamw_state_schema{"adamw_state", 1};

template <typename Tree>
[[nodiscard]] persistence::Record capture_optimizer_state(const ParameterSnapshot<Tree> &snapshot,
                                                          const AdamW::State &state)
{
    const std::size_t size = snapshot.parameters().size();
    std::vector<ParameterId> parameter_ids;
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    parameter_ids.reserve(size);
    index_by_id.reserve(size);

    persistence::Record paths;
    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, ParameterId id) {
        const auto [position, inserted] = index_by_id.try_emplace(id, parameter_ids.size());
        if (inserted)
        {
            parameter_ids.push_back(id);
        }
        paths.add(path, static_cast<std::uint64_t>(position->second));
    });

    persistence::Value::List moments;
    moments.reserve(state.moments_.size());
    for (std::size_t index = 0; index < parameter_ids.size(); ++index)
    {
        const ParameterId id = parameter_ids[index];
        const ParameterState &parameter = snapshot.get(id);
        const auto moment = state.moments_.find(id);
        if (!parameter.metadata.trainable)
        {
            if (moment != state.moments_.end())
            {
                throw std::invalid_argument{"AdamW moments reference a frozen parameter"};
            }
            continue;
        }
        if (moment == state.moments_.end())
        {
            throw std::invalid_argument{"AdamW state is missing parameter moments"};
        }

        const Tensor &value = parameter.value;
        const auto &[first, second] = moment->second;
        if (!is_floating_point(value.dtype()))
        {
            throw std::invalid_argument{"AdamW requires floating-point trainable parameters"};
        }
        if (first.shape() != value.shape() || first.dtype() != value.dtype() || first.device() != value.device() ||
            second.shape() != value.shape() || second.dtype() != value.dtype() || second.device() != value.device())
        {
            throw std::invalid_argument{"AdamW moments do not match parameter specification"};
        }

        persistence::Record entry;
        entry.add("parameter", static_cast<std::uint64_t>(index));
        entry.add("first", first);
        entry.add("second", second);
        moments.emplace_back(std::move(entry));
    }
    if (moments.size() != state.moments_.size())
    {
        throw std::invalid_argument{"AdamW state contains unexpected parameter moments"};
    }

    persistence::Record record = persistence::detail::make_object_record(adamw_state_schema);
    record.add("paths", std::move(paths));
    record.add("step", state.step());
    record.add("moments", std::move(moments));
    return record;
}

template <typename Tree>
[[nodiscard]] AdamW::State restore_adamw_state(const persistence::Record &record, const ParameterSnapshot<Tree> &target)
{
    persistence::detail::validate_object_record(record, adamw_state_schema, {"paths", "step", "moments"});

    const auto &paths = record.get("paths").get<persistence::Record>();
    std::unordered_map<ParameterId, std::size_t> target_to_index;
    target_to_index.reserve(target.parameters().size());
    std::vector<std::optional<ParameterId>> index_to_target(target.parameters().size());

    std::size_t matched_paths = 0;
    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, ParameterId id) {
        if (!paths.contains(path))
        {
            throw std::invalid_argument{"AdamW state is missing a target path"};
        }
        const auto saved_index = paths.get(path).get<std::uint64_t>();
        if (saved_index >= index_to_target.size())
        {
            throw std::out_of_range{"AdamW parameter path index is out of range"};
        }
        const auto index = static_cast<std::size_t>(saved_index);
        const auto [position, inserted] = target_to_index.try_emplace(id, index);
        if (!inserted && position->second != index)
        {
            throw std::invalid_argument{"tied target parameter references distinct saved indices"};
        }
        auto &target_id = index_to_target[index];
        if (target_id && *target_id != id)
        {
            throw std::invalid_argument{"tied saved parameter references distinct target parameters"};
        }
        target_id = id;
        ++matched_paths;
    });

    if (matched_paths != paths.size())
    {
        throw std::invalid_argument{"AdamW state contains unexpected paths"};
    }
    for (const auto &id : index_to_target)
    {
        if (!id)
        {
            throw std::invalid_argument{"AdamW saved parameter index has no target"};
        }
    }

    AdamW::State state;
    state.step_ = record.get("step").get<std::uint64_t>();
    const auto &moments = record.get("moments").get<persistence::Value::List>();
    state.moments_.reserve(target.parameters().size());
    for (const auto &saved : moments)
    {
        const auto &entry = saved.get<persistence::Record>();
        if (!entry.contains({"parameter", "first", "second"}, true))
        {
            throw std::invalid_argument{"AdamW moment record does not contain the expected fields"};
        }
        const auto saved_index = entry.get("parameter").get<std::uint64_t>();
        if (saved_index >= index_to_target.size())
        {
            throw std::out_of_range{"AdamW parameter reference is out of range"};
        }
        const ParameterId id = *index_to_target[static_cast<std::size_t>(saved_index)];
        const ParameterState &parameter = target.get(id);
        if (!parameter.metadata.trainable)
        {
            throw std::invalid_argument{"AdamW moments reference a frozen parameter"};
        }

        const Tensor &value = parameter.value;
        const auto &first = entry.get("first").get<Tensor>();
        const auto &second = entry.get("second").get<Tensor>();
        if (!is_floating_point(value.dtype()))
        {
            throw std::invalid_argument{"AdamW requires floating-point trainable parameters"};
        }
        if (first.shape() != value.shape() || first.dtype() != value.dtype() || first.device() != value.device() ||
            second.shape() != value.shape() || second.dtype() != value.dtype() || second.device() != value.device())
        {
            throw std::invalid_argument{"AdamW moments do not match parameter specification"};
        }
        if (!state.moments_.emplace(id, AdamW::State::Moment{first, second}).second)
        {
            throw std::invalid_argument{"AdamW state contains duplicate parameter moments"};
        }
    }

    for (const auto &[id, parameter] : target.parameters())
    {
        if (parameter.metadata.trainable && !state.moments_.contains(id))
        {
            throw std::invalid_argument{"AdamW state is missing parameter moments"};
        }
    }
    return state;
}
} // namespace minitensor::nn
