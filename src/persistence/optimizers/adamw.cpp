#include "minitensor/persistence/detail/parameter_binding.hpp"
#include <minitensor/persistence/detail/optimizers/adamw.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/nn/optimizers/detail/access/adamw.hpp>
#include <minitensor/types.hpp>

namespace minitensor::persistence::detail
{
namespace
{
using StateAccess = nn::detail::AdamWStateAccess;

// Validates AdamW moments against their corresponding parameter.
void validate_moments(const nn::ParameterState &parameter, const Tensor &first, const Tensor &second)
{
    if (!parameter.metadata.trainable)
    {
        throw std::invalid_argument{"AdamW moments reference a frozen parameter"};
    }
    const Tensor &value = parameter.value;
    if (!is_floating_point(value.dtype()))
    {
        throw std::invalid_argument{"AdamW requires floating-point trainable parameters"};
    }
    if (first.shape() != value.shape() || first.dtype() != value.dtype() || first.device() != value.device() ||
        second.shape() != value.shape() || second.dtype() != value.dtype() || second.device() != value.device())
    {
        throw std::invalid_argument{"AdamW moments do not match parameter specification"};
    }
}

// Verifies that all trainable parameters have been seen.
void validate_trainable_coverage(std::span<const nn::ParameterId> ids, const std::vector<bool> &seen,
                                 const nn::ParameterMap &parameters)
{
    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        if (parameters.get(ids[index]).metadata.trainable && !seen[index])
        {
            throw std::invalid_argument{"AdamW state is missing parameter moments"};
        }
    }
}
} // namespace

Record OptimizerPersistence<nn::AdamW>::capture_config(const nn::AdamW &optimizer)
{
    Record record;
    record.add("kind", "adamw_config");
    record.add("version", std::uint64_t{1});
    record.add("learning_rate", optimizer.learning_rate());
    record.add("weight_decay", optimizer.weight_decay());
    record.add("beta1", optimizer.beta1());
    record.add("beta2", optimizer.beta2());
    record.add("epsilon", optimizer.epsilon());
    return record;
}

nn::AdamW OptimizerPersistence<nn::AdamW>::restore_config(const Record &record)
{
    if (!record.contains({"kind", "version", "learning_rate", "weight_decay", "beta1", "beta2", "epsilon"}, true))
    {
        throw std::invalid_argument{"AdamW configuration record does not contain required fields"};
    }
    if (record.get("kind").get<std::string>() != "adamw_config")
    {
        throw std::invalid_argument{"unexpected AdamW record kind"};
    }
    if (record.get("version").get<std::uint64_t>() != 1)
    {
        throw std::invalid_argument{"unsupported AdamW record version"};
    }

    return nn::AdamW{record.get("learning_rate").get<float>(), record.get("weight_decay").get<float>(),
                     record.get("beta1").get<float>(), record.get("beta2").get<float>(),
                     record.get("epsilon").get<float>()};
}

Record OptimizerPersistence<nn::AdamW>::capture_state(const nn::AdamW::State &state, const nn::ParameterMap &parameters,
                                                      const ParameterBinding &binding)
{
    // The caller creates the mapping from the same snapshot as parameters.
    std::vector<bool> seen(binding.parameter_ids.size(), false);
    std::vector<std::pair<std::size_t, Record>> entries;
    entries.reserve(parameters.size());
    StateAccess::for_each_moment(state, [&](nn::ParameterId id, const Tensor &first, const Tensor &second) {
        const auto position = binding.index_by_id.find(id);
        if (position == binding.index_by_id.end())
        {
            throw std::invalid_argument{"AdamW state contains unexpected parameter moments"};
        }
        validate_moments(parameters.get(id), first, second);
        const std::size_t index = position->second;
        seen[index] = true;

        Record entry;
        entry.add("parameter", static_cast<std::uint64_t>(index));
        entry.add("first", first);
        entry.add("second", second);
        entries.emplace_back(index, std::move(entry));
    });
    validate_trainable_coverage(binding.parameter_ids, seen, parameters);

    // State traversal is unordered; emit moments in saved-index order.
    std::sort(entries.begin(), entries.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    Value::List moments;
    moments.reserve(entries.size());
    for (auto &entry : entries)
    {
        moments.emplace_back(std::move(entry.second));
    }

    Record record;
    record.add("kind", "adamw_state");
    record.add("version", std::uint64_t{1});
    record.add("step", state.step());
    record.add("moments", std::move(moments));
    return record;
}

nn::AdamW::State OptimizerPersistence<nn::AdamW>::restore_state(const Record &record,
                                                                const nn::ParameterMap &parameters,
                                                                std::span<const nn::ParameterId> parameter_ids)
{
    // Validate header fields.
    if (!record.contains({"kind", "version", "step", "moments"}, true))
    {
        throw std::invalid_argument{"AdamW state record does not contain required fields"};
    }
    if (record.get("kind").get<std::string>() != "adamw_state")
    {
        throw std::invalid_argument{"unexpected AdamW record kind"};
    }
    if (record.get("version").get<std::uint64_t>() != 1)
    {
        throw std::invalid_argument{"unsupported AdamW record version"};
    }

    const auto step = record.get("step").get<std::uint64_t>();

    std::vector<bool> seen(parameter_ids.size(), false);
    const auto &saved_moments = record.get("moments").get<Value::List>();
    std::vector<StateAccess::MomentEntry> moments;
    moments.reserve(saved_moments.size());
    for (const Value &saved : saved_moments)
    {
        // Confirm entry validity.
        const auto &entry = saved.get<Record>();
        if (!entry.contains({"parameter", "first", "second"}, true))
        {
            throw std::invalid_argument{"AdamW moment record does not contain required fields"};
        }

        // Obtain record index.
        const auto saved_index = entry.get("parameter").get<std::uint64_t>();
        if (saved_index >= parameter_ids.size())
        {
            throw std::out_of_range{"AdamW parameter reference is out of range"};
        }
        const auto index = static_cast<std::size_t>(saved_index);

        // Reject duplicate moments.
        if (seen[index])
        {
            throw std::invalid_argument{"AdamW state contains duplicate parameter moments"};
        }
        const nn::ParameterId id = parameter_ids[index];
        const Tensor &first = entry.get("first").get<Tensor>();
        const Tensor &second = entry.get("second").get<Tensor>();
        validate_moments(parameters.get(id), first, second);
        seen[index] = true;
        moments.push_back(StateAccess::MomentEntry{id, first, second});
    }
    validate_trainable_coverage(parameter_ids, seen, parameters);
    return StateAccess::make(step, moments);
}
} // namespace minitensor::persistence::detail
