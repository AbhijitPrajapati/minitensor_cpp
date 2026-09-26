#pragma once

#include <span>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
	[[nodiscard]] Tensor contiguous(const Tensor& input);
	[[nodiscard]] Tensor permute(const Tensor& input, std::span<const Axis> permutation);
	[[nodiscard]] Tensor transpose(const Tensor& input);
	[[nodiscard]] Tensor transpose(const Tensor& input, Axis axis0, Axis axis1);
	[[nodiscard]] Tensor reshape(const Tensor& input, Shape shape);
	[[nodiscard]] Tensor flatten(const Tensor& input);
	[[nodiscard]] Tensor flatten(const Tensor& input, Axis start_axis, Axis end_axis = -1);
	[[nodiscard]] Tensor squeeze(const Tensor& input);
	[[nodiscard]] Tensor squeeze(const Tensor& input, Axis axis);
	[[nodiscard]] Tensor squeeze(const Tensor& input, std::span<const Axis> axes);
	[[nodiscard]] Tensor unsqueeze(const Tensor& input, Axis axis);
	[[nodiscard]] Tensor broadcast_to(const Tensor& input, Shape shape);
}
