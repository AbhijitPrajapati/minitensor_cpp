#pragma once

#include <array>
#include <cstdint>

namespace minitensor
{
	namespace detail
	{
		// Accessor for random key words
		struct RandomKeyAccess;
	}

	class RandomKey final
	{
	public:
		explicit constexpr RandomKey(std::uint64_t seed) noexcept
			: words_{ seed, 0 }
		{}

		RandomKey(const RandomKey&) noexcept = default;
		RandomKey(RandomKey&&) noexcept = default;
		RandomKey& operator=(const RandomKey&) noexcept = default;
		RandomKey& operator=(RandomKey&&) noexcept = default;
		friend constexpr bool operator==(const RandomKey&, const RandomKey&) = default;

	private:
		constexpr RandomKey(std::uint64_t first, std::uint64_t second) noexcept
			: words_{ first, second }
		{}

		friend struct detail::RandomKeyAccess;
		friend RandomKey fold_in(RandomKey key, std::uint64_t data) noexcept;

		std::array<std::uint64_t, 2> words_;
	};

	[[nodiscard]] std::array<RandomKey, 2> split(RandomKey key) noexcept;
	[[nodiscard]] RandomKey fold_in(RandomKey key, std::uint64_t data) noexcept;
}
