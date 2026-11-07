#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn::detail
{

// Assign one index per unique parameter, even when the codec emits no payload for it.
template <typename Tree, typename Function>
[[nodiscard]] persistence::Record capture_parameter_paths(const ParameterSnapshot<Tree> &snapshot, Function &&function)
{
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    index_by_id.reserve(snapshot.parameters().size());

    persistence::Record paths;
    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, ParameterId id) {
        const auto [position, inserted] = index_by_id.try_emplace(id, index_by_id.size());
        if (inserted)
        {
            std::invoke(function, position->second, id, snapshot.get(id));
        }
        paths.add(path, static_cast<std::uint64_t>(position->second));
    });
    return paths;
}

struct ParameterBinding
{
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    std::vector<std::optional<ParameterId>> id_by_index;
};

template <typename Tree>
[[nodiscard]] ParameterBinding bind_parameter_paths(const persistence::Record &paths,
                                                    const ParameterSnapshot<Tree> &target, std::size_t index_count)
{
    ParameterBinding binding;
    binding.index_by_id.reserve(target.parameters().size());
    binding.id_by_index.resize(index_count);

    std::size_t matched_paths = 0;
    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, ParameterId id) {
        if (!paths.contains(path))
        {
            throw std::invalid_argument{"saved parameter layout is missing a target path"};
        }
        const auto saved_index = paths.get(path).get<std::uint64_t>();
        if (saved_index >= index_count)
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
    if (binding.index_by_id.size() != index_count)
    {
        throw std::invalid_argument{"saved parameter index has no target"};
    }
    return binding;
}
} // namespace minitensor::nn::detail
