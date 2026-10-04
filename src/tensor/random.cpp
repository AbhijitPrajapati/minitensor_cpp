#include <minitensor/random.hpp>

#include <array>
#include <cstdint>

namespace minitensor
{
	namespace
	{
		constexpr std::uint64_t golden_ratio = 0x9E3779B97F4A7C15ULL;
		constexpr std::uint64_t split_tag = 0xD2B74407B1CE6E93ULL;

		std::uint64_t mix(std::uint64_t value) noexcept
		{
			value += golden_ratio;
			value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
			value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
			return value ^ (value >> 31);
		}
	}

	std::array<RandomKey, 2> split(RandomKey key) noexcept
	{
		return {
			fold_in(key, split_tag),
			fold_in(key, split_tag + 1)
		};
	}

	RandomKey fold_in(RandomKey key, std::uint64_t data) noexcept
	{
		const std::uint64_t first_data = mix(data);
		const std::uint64_t second_data = mix(data ^ golden_ratio);
		return RandomKey{
			mix(key.words_[0] ^ first_data),
			mix(key.words_[1] ^ second_data)
		};
	}
}
