#pragma once

#include <array>
#include <cstdint>

#include <minitensor/random.hpp>

namespace minitensor::detail
{
	struct RandomKeyAccess final
	{
		[[nodiscard]] static constexpr const std::array<std::uint64_t, 2>& words(
			const RandomKey& key) noexcept
		{
			return key.words_;
		}
	};
}
