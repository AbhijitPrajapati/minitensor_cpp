#pragma once

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
	[[nodiscard]] Tensor full(Shape shape, float value, TensorOptions options = {});
	[[nodiscard]] Tensor full_like(const Tensor& input, float value);
	[[nodiscard]] Tensor full_like(const Tensor& input, float value, TensorOptions options);
	[[nodiscard]] Tensor zeros(Shape shape, TensorOptions options = {});
	[[nodiscard]] Tensor ones(Shape shape, TensorOptions options = {});
	[[nodiscard]] Tensor zeros_like(const Tensor& input);
	[[nodiscard]] Tensor zeros_like(const Tensor& input, TensorOptions options);
	[[nodiscard]] Tensor ones_like(const Tensor& input);
	[[nodiscard]] Tensor ones_like(const Tensor& input, TensorOptions options);
	[[nodiscard]] Tensor uniform(Shape shape, float low, float high, TensorOptions options = {});
	[[nodiscard]] Tensor normal(Shape shape, float mean, float std_dev, TensorOptions options = {});
	[[nodiscard]] Tensor uniform_like(const Tensor& input, float low, float high);
	[[nodiscard]] Tensor uniform_like(const Tensor& input, float low, float high, TensorOptions options);
	[[nodiscard]] Tensor normal_like(const Tensor& input, float mean, float std_dev);
	[[nodiscard]] Tensor normal_like(const Tensor& input, float mean, float std_dev, TensorOptions options);
}
