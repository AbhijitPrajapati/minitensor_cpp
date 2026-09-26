#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
	class ConcatenatePrimitive final : public Primitive
	{
	public:
		explicit ConcatenatePrimitive(Axis axis, Shape::size_type input_rank);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] Shape::size_type axis() const noexcept;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor& output_cotangent) const override;

	private:
		Shape::size_type axis_;
		Shape::size_type input_rank_;
	};
}
