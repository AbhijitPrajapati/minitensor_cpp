#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include <minitensor/tensor.hpp>


namespace minitensor::detail
{
	class Primitive;

	[[nodiscard]] Tensor apply_operation(
		std::unique_ptr<Primitive> primitive,
		std::span<const Tensor> inputs);

	template <typename... Inputs>
		requires (std::same_as<std::remove_cvref_t<Inputs>, Tensor> && ...)
	[[nodiscard]] Tensor apply_operation(
		std::unique_ptr<Primitive> primitive,
		const Inputs&... inputs)
	{
		const std::array<Tensor, sizeof...(Inputs)> input_tensors{ inputs... };
		return apply_operation(std::move(primitive), input_tensors);
	}
}
