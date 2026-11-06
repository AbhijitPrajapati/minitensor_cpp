#include <algorithm>

#include "minitensor/persistence/detail/parameter_binding.hpp"
#include "minitensor/persistence/detail/parameters.hpp"
#include "minitensor/persistence/record.hpp"

namespace minitensor::persistence::detail
{

ParsedParameters parse_parameters(const Record &record)
{
    // Validates header fields.
    if (!record.contains({"kind", "version", "paths", "states"}, true))
    {
        throw std::invalid_argument{"snapshot record does not contain required fields"};
    }

    // Confirm kind and version.
    const auto &kind = record.get("kind").get<std::string>();
    if (kind != "parameter_snapshot")
    {
        throw std::invalid_argument{"record is not a parameter snapshot"};
    }
    const auto version = record.get("version").get<std::uint64_t>();
    if (version != 1)
    {
        throw std::invalid_argument{"unsupported parameter snapshot version"};
    }

    // Parse saved states upfront into ParameterState
    const auto &saved_states = record.get("states").get<Value::List>();
    std::vector<nn::ParameterState> states;
    states.reserve(saved_states.size());

    for (const persistence::Value &entry : saved_states)
    {
        const auto &state_record = entry.get<persistence::Record>();
        if (!state_record.contains({"value", "trainability"}, true))
        {
            throw std::invalid_argument{"state record does not contain required fields"};
        }
        const auto &value = state_record.get("value").get<Tensor>();
        const bool trainability = state_record.get("trainability").get<bool>();
        states.push_back(nn::ParameterState{value, nn::ParameterMetadata{trainability}});
    }

    ParameterReferences references;
    references.state_count = states.size();

    // Collect path-index mappings, ensuring that each state is referenced at least once.
    const auto &paths = record.get("paths").get<Record>();
    std::vector<bool> referenced(states.size(), false);
    for (const auto &[path, entry] : paths)
    {
        const auto saved_index = entry.get<std::uint64_t>();
        if (saved_index >= states.size())
        {
            throw std::out_of_range{"snapshot state index is out of range"};
        }

        const auto index = static_cast<std::size_t>(saved_index);
        references.index_by_path.emplace(path, index);
        referenced[index] = true;
    }

    if (std::find(referenced.begin(), referenced.end(), false) != referenced.end())
    {
        throw std::invalid_argument{"snapshot contains unreferenced state"};
    }

    return {std::move(references), std::move(states)};
}
} // namespace minitensor::persistence::detail