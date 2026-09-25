#pragma once

#include <array>
#include <cstdint>

#include "tensor/core/random.hpp"

namespace minitensor::detail::cpu
{
    struct RandomBlock final
    {
        std::array<std::uint32_t, 4> words;
    };

    [[nodiscard]] RandomBlock random_block(RandomKey key, std::uint64_t block_idx) noexcept;
    // [0, 1), with 24 bits of precision suitable for Float32
    [[nodiscard]] float unit_uniform(std::uint32_t bits) noexcept;
    // (0, 1), retaining all 32 source bits in a double
    [[nodiscard]] double open_unit_uniform(std::uint32_t bits) noexcept;
}