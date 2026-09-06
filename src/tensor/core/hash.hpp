#pragma once

#include <cstddef>
#include <functional>

namespace minitensor::detail
{
    template <typename T>
    inline void combine_hash(std::size_t &seed, const T &value) noexcept(noexcept(std::hash<T>{}(value)))
    {
        const std::size_t value_hash = std::hash<T>{}(value);
        seed ^= value_hash + std::size_t{0x9e3779b9} + (seed << 6) + (seed >> 2);
    }
}