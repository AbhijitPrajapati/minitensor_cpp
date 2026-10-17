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
	class GatherPrimitive final : public Primitive
	{
	public:
		static constexpr PrimitiveId ID{ 0x0208 };
		[[nodiscard]] PrimitiveId id() const noexcept override { return ID; }
		GatherPrimitive(Axis axis, Shape::size_type input_rank);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] Shape::size_type axis() const noexcept;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor& output_cotangent) const override;
	private:
		Shape::size_type axis_;
		Shape::size_type input_rank_;
	};

	// Gather and GatherScatterAdd are eachother's VJPs

	class GatherScatterAddPrimitive final : public Primitive
	{
	public:
		static constexpr PrimitiveId ID{ 0x0209 };
		[[nodiscard]] PrimitiveId id() const noexcept override { return ID; }
		GatherScatterAddPrimitive(Shape target_shape, Shape::size_type axis);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] Shape::size_type axis() const noexcept;
		[[nodiscard]] const Shape& target_shape() const noexcept;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor& output_cotangent) const override;
	private:
		Shape target_shape_;
		Shape::size_type axis_;
	};
}
