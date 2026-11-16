#pragma once

#include <cstdint>

#include <minitensor/random.hpp>

// Add CUDA compiler extensions for the CUDA backend.
#if defined(__CUDACC__)
#define MINITENSOR_RANDOM_HOST_DEVICE __host__ __device__
#else
#define MINITENSOR_RANDOM_HOST_DEVICE
#endif

namespace minitensor::detail
{
struct RandomBlock final
{
    static constexpr std::uint32_t word_count = 4;
    std::uint32_t words[word_count];
};

namespace philox
{
inline constexpr std::uint32_t multiplier0 = 0xD2511F53U;
inline constexpr std::uint32_t multiplier1 = 0xCD9E8D57U;
inline constexpr std::uint32_t weyl0 = 0x9E3779B9U;
inline constexpr std::uint32_t weyl1 = 0xBB67AE85U;
inline constexpr std::uint32_t round_count = 10;

[[nodiscard]] MINITENSOR_RANDOM_HOST_DEVICE inline RandomBlock philox_round(RandomBlock counter, std::uint32_t key0,
                                                                            std::uint32_t key1) noexcept
{
    const auto product0 = static_cast<std::uint64_t>(multiplier0) * counter.words[0];
    const auto product1 = static_cast<std::uint64_t>(multiplier1) * counter.words[2];
    const auto low0 = static_cast<std::uint32_t>(product0);
    const auto high0 = static_cast<std::uint32_t>(product0 >> 32);
    const auto low1 = static_cast<std::uint32_t>(product1);
    const auto high1 = static_cast<std::uint32_t>(product1 >> 32);
    return {{high1 ^ counter.words[1] ^ key0, low1, high0 ^ counter.words[3] ^ key1, low0}};
}
} // namespace philox

// Philox4x32-10. Counter b supplies logical sample positions 4b through 4b + 3.
[[nodiscard]] MINITENSOR_RANDOM_HOST_DEVICE inline RandomBlock random_block(std::uint64_t key_first,
                                                                            std::uint64_t key_second,
                                                                            std::uint64_t counter_index) noexcept
{
    RandomBlock counter{{static_cast<std::uint32_t>(counter_index), static_cast<std::uint32_t>(counter_index >> 32),
                         static_cast<std::uint32_t>(key_second), static_cast<std::uint32_t>(key_second >> 32)}};
    auto key0 = static_cast<std::uint32_t>(key_first);
    auto key1 = static_cast<std::uint32_t>(key_first >> 32);

    for (std::uint32_t round = 0; round < philox::round_count; ++round)
    {
        counter = philox::philox_round(counter, key0, key1);
        if (round + 1 != philox::round_count)
        {
            key0 += philox::weyl0;
            key1 += philox::weyl1;
        }
    }
    return counter;
}

// Returns [0, 1), with 24 bits of precision suitable for float 32.
[[nodiscard]] MINITENSOR_RANDOM_HOST_DEVICE inline float unit_uniform(std::uint32_t bits) noexcept
{
    constexpr float scale = 0x1.0p-24F;
    return static_cast<float>(bits >> 8) * scale;
}

// Returns (0, 1), retaining all 32 source bits in a double.
[[nodiscard]] MINITENSOR_RANDOM_HOST_DEVICE inline double open_unit_uniform(std::uint32_t bits) noexcept
{
    constexpr double scale = 0x1.0p-32;
    return (static_cast<double>(bits) + 0.5) * scale;
}
} // namespace minitensor::detail

#undef MINITENSOR_RANDOM_HOST_DEVICE
