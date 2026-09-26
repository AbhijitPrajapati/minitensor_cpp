#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	class SliceParameters final
	{
	public:
		SliceParameters(
			Axis axis,
			std::optional<Extent> start,
			std::optional<Extent> stop,
			Extent step,
			const Shape &input_shape);
		[[nodiscard]] Shape::size_type axis() const noexcept;
		[[nodiscard]] Extent start() const noexcept;
		[[nodiscard]] Extent length() const noexcept;
		[[nodiscard]] Extent step() const noexcept;
		[[nodiscard]] Shape::size_type input_rank() const noexcept;
		[[nodiscard]] Extent input_axis_extent() const noexcept;

	private:
		Shape::size_type axis_;
		Extent start_;
		Extent length_;
		Extent step_;
		Shape::size_type input_rank_;
		Extent input_axis_extent_;
	};

	class SlicePrimitive final : public Primitive
	{
	public:
		SlicePrimitive(
			Axis axis,
			std::optional<Extent> start,
			std::optional<Extent> stop,
			Extent step,
			const Shape &input_shape);
		explicit SlicePrimitive(SliceParameters parameters);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] bool requires_kernel_support() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec &, const Layout &input_layout, const TensorSpec &output_spec) const override;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const override;

	private:
		SliceParameters parameters_;
	};

	class SliceScatterPrimitive final : public Primitive
	{
	public:
		SliceScatterPrimitive(Shape output_shape, SliceParameters parameters);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] Shape::size_type axis() const noexcept;
		[[nodiscard]] Extent start() const noexcept;
		[[nodiscard]] Extent step() const noexcept;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &output, const Tensor &output_cotangent) const override;

	private:
		Shape output_shape_;
		Shape input_shape_;
		SliceParameters parameters_;
	};
}
