#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
	class ConcatenatePrimitive final : public Primitive
	{
	public:
		static constexpr PrimitiveId ID{ 0x0205 };
		[[nodiscard]] PrimitiveId id() const noexcept override { return ID; }
		explicit ConcatenatePrimitive(Axis axis, Shape::size_type input_rank);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] Shape::size_type axis() const noexcept;

	private:
		Shape::size_type axis_;
		Shape::size_type input_rank_;
	};
}
