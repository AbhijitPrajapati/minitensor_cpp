#pragma once

#include <cstdint>
#include <istream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::persistence::detail
{
struct ArchiveData
{
    std::vector<Tensor> tensors;
    // Bindings map names to tensor indices. Multiple names may reference a single tensor to encode tied parameters.
    std::map<std::string, std::uint64_t, std::less<>> bindings;
};

void write_archive(std::ostream &output, const ArchiveData &data);
[[nodiscard]] ArchiveData read_archive(std::istream &input, Device device);
} // namespace minitensor::persistence::detail