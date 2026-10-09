#pragma once

#include <initializer_list>
#include <span>
#include <vector>

#include <minitensor/tensor.hpp>

namespace minitensor
{
	void eval(const Tensor& tensor);
	void eval(std::span<const Tensor> tensors);
	void eval(std::initializer_list<Tensor> tensors);

	// eval merely materializes an existing graph value,
	// while leafify evaluates and returns a producer-free leaf.
	[[nodiscard]] Tensor leafify(const Tensor& tensor);
	[[nodiscard]] std::vector<Tensor> leafify(std::span<const Tensor> tensors);
}