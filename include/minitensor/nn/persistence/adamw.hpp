#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/persistence/detail/parameter_binding.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn
{

constexpr persistence::detail::RecordHeaderSchema adamw_state_schema{"adamw_state", 1};

template <typename Tree>
[[nodiscard]] persistence::Record capture_optimizer_state(const ParameterSnapshot<Tree> &snapshot,
                                                          const AdamW::State &state)
{
    persistence::Value::List moments;
    moments.reserve(state.moments_.size());

    persistence::Record paths = detail::capture_parameter_paths(
        snapshot, [&](std::size_t index, ParameterId id, const ParameterState &parameter) {
            const auto moment = state.moments_.find(id);
            if (!parameter.metadata.trainable)
            {
                if (moment != state.moments_.end())
                {
                    throw std::invalid_argument{"AdamW moments reference a frozen parameter"};
                }
                return;
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
        });

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
    const auto binding = detail::bind_parameter_paths(paths, target, target.parameters().size());

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
        if (saved_index >= binding.id_by_index.size())
        {
            throw std::out_of_range{"AdamW parameter reference is out of range"};
        }
        const ParameterId id = *binding.id_by_index[static_cast<std::size_t>(saved_index)];
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
