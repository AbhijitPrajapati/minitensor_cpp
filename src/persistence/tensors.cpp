#include "minitensor/persistence/tensors.hpp"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "minitensor/persistence/detail/archive.hpp"

namespace minitensor::persistence
{
void write_tensors(std::ostream &output, const NamedTensors &tensors)
{
    detail::ArchiveData data;

    const auto tensor_count = tensors.size();
    if (!std::in_range<std::uint64_t>(tensor_count))
    {
        throw std::overflow_error{"tensor count exceeds archive index range"};
    }
    data.tensors.reserve(tensor_count);

    // Capture a one-to-one binding between names and tensors.
    for (const auto &[name, tensor] : tensors)
    {
        data.bindings.emplace_hint(data.bindings.end(), name, static_cast<std::uint64_t>(data.tensors.size()));
        data.tensors.push_back(tensor);
    }

    detail::write_archive(output, data);
}

NamedTensors read_tensors(std::istream &input, Device device)
{
    detail::ArchiveData data = detail::read_archive(input, device);

    if (data.bindings.size() != data.tensors.size())
    {
        throw std::invalid_argument{"archive data does not contain a one-to-one tensor binding"};
    }

    NamedTensors output;

    // Track seen indices to enforce a one-to-one mapping.
    std::vector<bool> seen_indices(data.tensors.size(), false);

    for (const auto &[name, saved_index] : data.bindings)
    {
        if (std::cmp_greater_equal(saved_index, data.tensors.size()))
        {
            throw std::invalid_argument{"saved tensor binding references an unknown tensor"};
        }

        const auto index = static_cast<std::size_t>(saved_index);
        if (seen_indices[index])
        {
            throw std::invalid_argument{"multiple saved tensor bindings reference the same tensor"};
        }
        seen_indices[index] = true;
        output.emplace_hint(output.end(), name, std::move(data.tensors[index]));
    }

    return output;
}
} // namespace minitensor::persistence
