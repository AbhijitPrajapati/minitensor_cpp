#pragma once

#include <cstdint>
#include <vector>

namespace minitensor {

using Index = std::int64_t;
using Shape = std::vector<Index>;
using Strides = std::vector<Index>;
using Coordinates = std::vector<Index>;

} // namespace minitensor
