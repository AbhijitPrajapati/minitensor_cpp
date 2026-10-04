#include <minitensor/ops/creation.hpp>

#include <memory>
#include <utility>

#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "apply_primitive.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"

namespace minitensor
{
	namespace
	{
		TensorOptions options_like(const Tensor& tensor)
		{
			return TensorOptions{ tensor.dtype(), tensor.device() };
		}

		detail::TensorSpec output_spec(Shape shape, TensorOptions options)
		{
			return detail::TensorSpec{ std::move(shape), options.dtype, options.device };
		}
	}

	Tensor full(Shape shape, float value, TensorOptions options)
	{
		return detail::apply_primitive(
			std::make_unique<detail::FullPrimitive>(output_spec(std::move(shape), options), value));
	}

	Tensor full_like(const Tensor& input, float value)
	{
		return full(input.shape(), value, options_like(input));
	}

	Tensor full_like(const Tensor& input, float value, TensorOptions options)
	{
		return full(input.shape(), value, options);
	}

	Tensor zeros(Shape shape, TensorOptions options)
	{
		return full(std::move(shape), 0.0F, options);
	}

	Tensor ones(Shape shape, TensorOptions options)
	{
		return full(std::move(shape), 1.0F, options);
	}

	Tensor zeros_like(const Tensor& input)
	{
		return full_like(input, 0.0F);
	}

	Tensor zeros_like(const Tensor& input, TensorOptions options)
	{
		return full_like(input, 0.0F, options);
	}

	Tensor ones_like(const Tensor& input)
	{
		return full_like(input, 1.0F);
	}

	Tensor ones_like(const Tensor& input, TensorOptions options)
	{
		return full_like(input, 1.0F, options);
	}

	Tensor uniform(Shape shape, float low, float high, RandomKey key, TensorOptions options)
	{
		detail::TensorSpec spec = output_spec(std::move(shape), options);
		return detail::apply_primitive(
			std::make_unique<detail::UniformPrimitive>(std::move(spec), low, high, key));
	}

	Tensor normal(Shape shape, float mean, float std_dev, RandomKey key, TensorOptions options)
	{
		detail::TensorSpec spec = output_spec(std::move(shape), options);
		return detail::apply_primitive(
			std::make_unique<detail::NormalPrimitive>(std::move(spec), mean, std_dev, key));
	}

	Tensor uniform_like(const Tensor& input, float low, float high, RandomKey key)
	{
		return uniform(input.shape(), low, high, key, options_like(input));
	}

	Tensor uniform_like(const Tensor& input, float low, float high, RandomKey key, TensorOptions options)
	{
		return uniform(input.shape(), low, high, key, options);
	}

	Tensor normal_like(const Tensor& input, float mean, float std_dev, RandomKey key)
	{
		return normal(input.shape(), mean, std_dev, key, options_like(input));
	}

	Tensor normal_like(const Tensor& input, float mean, float std_dev, RandomKey key, TensorOptions options)
	{
		return normal(input.shape(), mean, std_dev, key, options);
	}
}
