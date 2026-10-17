#pragma once

#include <span>
#include <string_view>
#include <vector>
#include <optional>


#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
	class SubtractPrimitive final : public Primitive
	{
	public:
		static constexpr PrimitiveId ID{ 0x0111 };
		[[nodiscard]] PrimitiveId id() const noexcept override { return ID; }
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
	};
}
