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
	// Internal rank-preserving unit-stride slice used to split cotangents.
	class NarrowPrimitive final : public Primitive
	{
	public:
		NarrowPrimitive(Shape::size_type axis, Extent start, Extent length, Shape::size_type input_rank);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] bool requires_kernel_support() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec& input_spec, const Layout& input_layout, const TensorSpec& output_spec) const override;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor& output, const Tensor& output_cotangent) const override;

	private:
		Shape::size_type axis_;
		Extent start_;
		Extent length_;
		Shape::size_type input_rank_;
	};
}
