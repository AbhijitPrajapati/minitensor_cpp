#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/host_scalar.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
	class FullPrimitive final : public Primitive
	{
	public:
		static constexpr PrimitiveId ID{ 0x0001 };
		[[nodiscard]] PrimitiveId id() const noexcept override { return ID; }
		explicit FullPrimitive(TensorSpec output_spec, HostScalar fill_value);
		[[nodiscard]] std::string_view name() const noexcept override;
		[[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
		[[nodiscard]] const HostScalar& fill_value() const noexcept;

	private:
		TensorSpec output_spec_;
		HostScalar fill_value_;
	};
}
