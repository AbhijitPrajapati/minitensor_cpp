#pragma once

#include <istream>
#include <map>
#include <ostream>
#include <string>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::persistence
{
using NamedTensors = std::map<std::string, Tensor, std::less<>>;

void write_tensors(std::ostream &output, const NamedTensors &tensors);
[[nodiscard]] NamedTensors read_tensors(std::istream &input, Device device = Device::cpu());
} // namespace minitensor::persistence
