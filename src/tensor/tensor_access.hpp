#pragma once

#include <utility>

#include <minitensor/tensor.hpp>

#include "graph/fwd.hpp"

namespace minitensor::detail
{
	// TensorAccess is the interface between the public Tensor handle and 
	// the internal graph Value.
	struct TensorAccess final
	{
		[[nodiscard]] static const ValueRef& value(const Tensor& tensor) noexcept
		{
			return tensor.value_;
		}

		[[nodiscard]] static Tensor make(ValueRef value)
		{
			return Tensor{ std::move(value) };
		}
	};
}
