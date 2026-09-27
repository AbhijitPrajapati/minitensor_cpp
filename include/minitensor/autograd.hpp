#pragma once

#include <initializer_list>
#include <span>
#include <vector>

#include <minitensor/tensor.hpp>

namespace minitensor
{
	[[nodiscard]] std::vector<Tensor> vjp(const Tensor& output, std::span<const Tensor> inputs, const Tensor& output_cotangent);
	[[nodiscard]] std::vector<Tensor> vjp(const Tensor& output, std::initializer_list<Tensor> inputs, const Tensor& output_cotangent);
	[[nodiscard]] std::vector<Tensor> vjp(const Tensor& output, const Tensor& input, const Tensor& output_cotangent);
	[[nodiscard]] std::vector<Tensor> grad(const Tensor& output, std::span<const Tensor> inputs);
	[[nodiscard]] std::vector<Tensor> grad(const Tensor& output, std::initializer_list<Tensor> inputs);
	[[nodiscard]] std::vector<Tensor> grad(const Tensor& output, const Tensor& input);
}