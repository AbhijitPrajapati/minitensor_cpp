#pragma once

#include <cstdint>

namespace minitensor::detail::cuda
{
[[nodiscard]] std::uint32_t cuda_device_count();
}