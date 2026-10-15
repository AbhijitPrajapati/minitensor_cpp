#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <minitensor/types.hpp>

namespace minitensor::detail::cpu
{
	// These functions handle dtype dispatch, converting the dynamic DType value
	// into a type identity used by typed kernels. Multiple dispatch functions
	// reflect how not every operation supports every DType.

	template <typename Function>
	decltype(auto) dispatch_dtype(DType dtype, Function&& function)
	{
		switch (dtype)
		{
		case DType::Float32:
			return std::invoke(std::forward<Function>(function), std::type_identity<float>{});
		case DType::Int64:
			return std::invoke(
				std::forward<Function>(function),
				std::type_identity<std::int64_t>{});
		}
		throw std::logic_error{ "unsupported CPU dtype" };
	}

	template <typename Function>
	decltype(auto) dispatch_floating_dtype(DType dtype, Function&& function)
	{
		switch (dtype)
		{
		case DType::Float32:
			return std::invoke(std::forward<Function>(function), std::type_identity<float>{});
		case DType::Int64:
			break;
		}
		throw std::logic_error{ "CPU kernel requires a floating-point dtype" };
	}
}
