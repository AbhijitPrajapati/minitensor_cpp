#pragma once

#include <cstddef>
#include <span>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"

namespace minitensor::detail
{
	[[nodiscard]] ValueRef make_value_from_host(TensorSpec spec, std::span<const std::byte> source);
	void copy_value_to_host(std::span<std::byte> destination, const ValueRef& value);
}