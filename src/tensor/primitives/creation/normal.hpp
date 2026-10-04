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
	class NormalPrimitive final : public Primitive
	{
	public:
		explicit NormalPrimitive(TensorSpec output_spec, float mean, float std_dev, RandomKey key);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] float mean() const noexcept;
		[[nodiscard]] float std_dev() const noexcept;
		[[nodiscard]] const RandomKey& key() const noexcept;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor&) const override;

	private:
		TensorSpec output_spec_;
		float mean_;
		float std_dev_;
		RandomKey key_;
	};
}
