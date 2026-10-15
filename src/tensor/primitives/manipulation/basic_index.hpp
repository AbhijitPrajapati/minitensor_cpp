#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	// Parameters for basic indexing are encapsulated in an independent class
	// for brevity regarding shared logic/validation between BasicIndexPrimitive and 
	// its VJP.
	class BasicIndexParameters final
	{
	public:
		[[nodiscard]] static BasicIndexParameters for_slice(
			Axis axis,
			std::optional<Extent> start,
			std::optional<Extent> stop,
			Extent step,
			const Shape& input_shape);

		[[nodiscard]] static BasicIndexParameters for_select(Axis axis, Extent index, const Shape& input_shape);

		[[nodiscard]] const Shape& input_shape() const noexcept;
		[[nodiscard]] const Shape& output_shape() const noexcept;
		[[nodiscard]] Layout derive_layout(const Layout& input_layout) const;

	private:
		struct AllIndex final {};

		struct SliceIndex final
		{
			Extent start;
			Extent length;
			Extent step;
		};

		struct SelectIndex final
		{
			Extent index;
		};

		using Component = std::variant<AllIndex, SliceIndex, SelectIndex>;

		BasicIndexParameters(
			Shape input_shape,
			Shape output_shape,
			std::vector<Component> components);

		Shape input_shape_;
		Shape output_shape_;
		std::vector<Component> components_;
	};

	// This primitive is used for basic slicing and select operations where a 
	// view is returned. The gather operation has a separate primitive.
	class BasicIndexPrimitive final : public Primitive
	{
	public:
		explicit BasicIndexPrimitive(BasicIndexParameters parameters);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] bool requires_kernel_support() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec&, const Layout& input_layout, const TensorSpec&) const override;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor& output_cotangent) const override;

	private:
		BasicIndexParameters parameters_;
	};

	// BasicIndex and BasicIndexScatter are eachother's VJPs.

	class BasicIndexScatterPrimitive final : public Primitive
	{
	public:
		explicit BasicIndexScatterPrimitive(BasicIndexParameters parameters);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] const BasicIndexParameters& parameters() const noexcept;
		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor& output, const Tensor& output_cotangent) const override;

	private:
		BasicIndexParameters parameters_;
	};
}
