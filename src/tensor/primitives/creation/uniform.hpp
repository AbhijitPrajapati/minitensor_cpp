#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
	class UniformPrimitive final : public Primitive
	{
	public:
		explicit UniformPrimitive(TensorSpec output_spec, float low, float high, RandomKey key);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] float low() const noexcept;
		[[nodiscard]] float high() const noexcept;
		[[nodiscard]] const RandomKey& key() const noexcept;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor&) const override;

	private:
		TensorSpec output_spec_;
		float low_;
		float high_;
		RandomKey key_;
	};
}
