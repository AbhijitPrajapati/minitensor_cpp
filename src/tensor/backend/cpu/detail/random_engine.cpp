#include "random_engine.hpp"

#include <array>
#include <cstdint>

namespace minitensor::detail::cpu
{
    namespace
    {
        constexpr std::uint32_t multiplier0 = 0xD2511F53U;
        constexpr std::uint32_t multiplier1 = 0xCD9E8D57U;
        constexpr std::uint32_t weyl0 = 0x9E3779B9U;
        constexpr std::uint32_t weyl1 = 0xBB67AE85U;
        constexpr std::uint32_t round_count = 10;

        void philox_round(std::array<std::uint32_t, 4> &counter,
                          std::uint32_t key0,
                          std::uint32_t key1) noexcept
        {
            const auto product0 = static_cast<std::uint64_t>(multiplier0) * counter[0];
            const auto product1 = static_cast<std::uint64_t>(multiplier1) * counter[2];
            const auto low0 = static_cast<std::uint32_t>(product0);
            const auto high0 = static_cast<std::uint32_t>(product0 >> 32);
            const auto low1 = static_cast<std::uint32_t>(product1);
            const auto high1 = static_cast<std::uint32_t>(product1 >> 32);
            counter = {
                high1 ^ counter[1] ^ key0,
                low1,
                high0 ^ counter[3] ^ key1,
                low0};
        }
    }

    RandomBlock random_block(RandomKey key, std::uint64_t block_idx) noexcept
    {
        std::array<std::uint32_t, 4> counter{
            static_cast<std::uint32_t>(block_idx),
            static_cast<std::uint32_t>(block_idx >> 32),
            static_cast<std::uint32_t>(key.second),
            static_cast<std::uint32_t>(key.second >> 32)};

        auto key0 = static_cast<std::uint32_t>(key.first);
        auto key1 = static_cast<std::uint32_t>(key.first >> 32);

        for (std::uint32_t round = 0; round < round_count; ++round)
        {
            philox_round(counter, key0, key1);
            if (round + 1 != round_count)
            {
                key0 += weyl0;
                key1 += weyl1;
            }
        }
        return RandomBlock{counter};
    }

    float unit_uniform(std::uint32_t bits) noexcept
    {
        constexpr float scale = 0x1.0p-24F;
        return static_cast<float>(bits >> 8) * scale;
    }

    double open_unit_uniform(std::uint32_t bits) noexcept
    {
        constexpr double scale = 0x1.0p-32;
        return (static_cast<double>(bits) + 0.5) * scale;
    }
}
