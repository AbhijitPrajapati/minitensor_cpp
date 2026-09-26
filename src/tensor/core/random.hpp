#pragma once

#include <cstdint>
#include <mutex>

namespace minitensor::detail
{
	struct RandomKey final
	{
		std::uint64_t first;
		std::uint64_t second;

		friend bool operator==(const RandomKey&, const RandomKey&) = default;
	};

	class Generator final
	{
	public:
		explicit Generator(std::uint64_t seed);
		Generator(const Generator&) = delete;
		Generator& operator=(const Generator&) = delete;

		void manual_seed(std::uint64_t seed);
		[[nodiscard]] RandomKey reserve_key();

	private:
		std::mutex mutex_;
		std::uint64_t seed_;
		std::uint64_t next_stream_;
		bool exhausted_{ false };
	};
}
