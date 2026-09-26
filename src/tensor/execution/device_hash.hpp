#pragma once

#include <minitensor/types.hpp>

#include "tensor/core/hash.hpp"

namespace minitensor::detail
{
	struct DeviceHash final
	{
		[[nodiscard]] std::size_t operator()(const Device& device) const noexcept
		{
			std::size_t result = 0;
			combine_hash(result, device.type());
			combine_hash(result, device.index());
			return result;
		}
	};
}