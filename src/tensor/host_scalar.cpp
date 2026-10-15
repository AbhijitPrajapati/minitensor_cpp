#include <minitensor/host_scalar.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>

#include <minitensor/types.hpp>

namespace minitensor
{
	HostScalar::HostScalar(float value) noexcept : value_{ value }
	{}

	HostScalar::HostScalar(std::int64_t value) noexcept : value_{ value }
	{}

	DType HostScalar::dtype() const noexcept
	{
		return std::holds_alternative<float>(value_)
			? DType::Float32
			: DType::Int64;
	}

	HostScalar HostScalar::cast(DType dtype) const
	{
		switch (dtype)
		{
		case DType::Float32:
			if (const float* const value = std::get_if<float>(&value_))
			{
				return HostScalar{ *value };
			}
			return HostScalar{ static_cast<float>(std::get<std::int64_t>(value_)) };

		case DType::Int64:
			if (const std::int64_t* const value = std::get_if<std::int64_t>(&value_))
			{
				return HostScalar{ *value };
			}

			const float value = std::get<float>(value_);

			// Range check before casting float to int
			if (!std::isfinite(value))
			{
				throw std::invalid_argument{ "cannot convert a non-finite scalar to int64" };
			}
			const long double widened = static_cast<long double>(value);
			if (widened < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
				widened > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
			{
				throw std::out_of_range{ "floating scalar is outside the int64 range" };
			}
			return HostScalar{ static_cast<std::int64_t>(value) };
		}
		throw std::invalid_argument{ "unrecognized dtype" };
	}
}
