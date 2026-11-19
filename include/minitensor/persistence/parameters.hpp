#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/parameters/parameter_tree.hpp>
#include <minitensor/types.hpp>

#include "detail/archive.hpp"

namespace minitensor::persistence
{

template <typename Tree>
void write_parameter_values(std::ostream &output, const nn::ParameterSnapshot<Tree> &snapshot)
{
    detail::ArchiveData data;

    // Validate and reserve parameter count.
    const auto parameter_count = snapshot.parameters().size();
    if (!std::in_range<std::uint64_t>(parameter_count))
    {
        throw std::overflow_error{"parameter count exceeds archive index range"};
    }
    data.tensors.reserve(parameter_count);

    // Each unique parameter has a file-local index.
    std::unordered_map<nn::ParameterId, std::size_t> index_by_id;
    index_by_id.reserve(parameter_count);

    nn::for_each_named_parameter(snapshot.tree(), [&](std::string_view path, nn::ParameterId id) {
        const auto [position, inserted] = index_by_id.try_emplace(id, index_by_id.size());
        if (inserted)
        {
            // Add a distinct tensor if a parameter ID is unseen.
            data.tensors.emplace_back(snapshot.value(id));
        }
        // Register the path-index correspondence for tied parameters.
        data.bindings.emplace(path, static_cast<std::uint64_t>(position->second));
    });
    detail::write_archive(output, data);
}

template <typename Tree>
[[nodiscard]] nn::ParameterSnapshot<Tree> read_parameter_values(std::istream &input, nn::ParameterSnapshot<Tree> target,
                                                                Device device = Device::cpu())
{
    detail::ArchiveData data = detail::read_archive(input, device);

    // Enforce a bijection between saved tensor indices and target IDs.
    std::unordered_map<nn::ParameterId, std::size_t> index_by_id;
    index_by_id.reserve(target.parameters().size());
    std::unordered_map<std::size_t, nn::ParameterId> id_by_index;
    id_by_index.reserve(target.parameters().size());
    std::size_t matched_paths = 0;

    nn::for_each_named_parameter(target.tree(), [&](std::string_view path, nn::ParameterId id) {
        const auto binding = data.bindings.find(path);
        if (binding == data.bindings.end())
        {
            throw std::invalid_argument{"saved parameter layout is missing a target path"};
        }
        if (std::cmp_greater_equal(binding->second, data.tensors.size()))
        {
            throw std::invalid_argument{"saved parameter binding references an unknown tensor"};
        }

        const auto index = static_cast<std::size_t>(binding->second);

        // Verify a one-to-one mapping between saved indices and target IDs.
        const auto [id_position, new_id] = index_by_id.try_emplace(id, index);
        if (!new_id && id_position->second != index)
        {
            throw std::invalid_argument{"shared target parameter has different saved tensor indices"};
        }
        const auto [index_position, new_index] = id_by_index.try_emplace(index, id);
        if (!new_index && index_position->second != id)
        {
            throw std::invalid_argument{"shared saved tensor is bound to different target parameters"};
        }
        if (new_id)
        {
            // Verify that tensor specifications match.
            const Tensor &expected = target.value(id);
            const Tensor &value = data.tensors[index];
            if (value.shape() != expected.shape() || value.dtype() != expected.dtype())
            {
                throw std::invalid_argument{"saved tensor shape or dtype does not match target parameter"};
            }
        }
        ++matched_paths;
    });

    // Ensure that every path is accounted for and every value slot is bound.
    if (matched_paths != data.bindings.size())
    {
        throw std::invalid_argument{"saved parameter layout contains unexpected paths"};
    }
    if (id_by_index.size() != data.tensors.size())
    {
        throw std::invalid_argument{"saved parameter tensor has no target"};
    }

    // Transform the target snapshot after validation.
    return std::move(target).transform([&](nn::ParameterId id, const nn::ParameterState &state) {
        return nn::ParameterState{std::move(data.tensors[index_by_id.at(id)]), state.metadata};
    });
}
} // namespace minitensor::persistence
