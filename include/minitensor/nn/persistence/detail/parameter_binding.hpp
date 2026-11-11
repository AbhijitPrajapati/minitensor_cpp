#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include <minitensor/nn/parameters/parameter_id.hpp>

namespace minitensor::nn::detail
{
// A bidirectional mapping between process-specific parameter IDs and record indices. Snapshot restoration constructs
// this binding using path correspondence,a and optimizer state restoration resuses it.
struct ParameterBinding
{
    std::unordered_map<ParameterId, std::size_t> index_by_id;
    // The optional vector allows for reservation despite the lack of default instantiations for ParameterId.
    std::vector<std::optional<ParameterId>> id_by_index;
};
} // namespace minitensor::nn::detail
