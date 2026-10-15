#pragma once

#include <memory>

#include "types.hpp"

namespace minitensor
{
	namespace detail
	{
		class Value;
		struct TensorAccess;
	} // namespace detail

	// The public Tensor handle is a thin wrapper around a Value
	// in the graph. This design faciliates natural Tensor semantics safely.
	class Tensor
	{
	public:
		Tensor(const Tensor&) noexcept = default;
		Tensor(Tensor&&) noexcept = default;
		Tensor& operator=(const Tensor&) noexcept = default;
		Tensor& operator=(Tensor&&) noexcept = default;

		[[nodiscard]] const Shape& shape() const noexcept;
		[[nodiscard]] std::size_t rank() const noexcept;
		[[nodiscard]] std::size_t numel() const noexcept;
		[[nodiscard]] DType dtype() const noexcept;
		[[nodiscard]] Device device() const noexcept;

	private:
		explicit Tensor(std::shared_ptr<detail::Value>);
		std::shared_ptr<detail::Value> value_;

		// This centralized friend structure allows operations to accces
		// the Value of Tensors and create Tensors from Values.
		friend struct detail::TensorAccess;
	};

} // namespace minitensor
