#pragma once

#include <minitensor/types.hpp>

namespace minitensor::detail
{

	// Tensor views are the way in which tensor data is represented to backend kernels.
	// Mutable tensor views are used for output tensors.

	struct TensorSpec;
	class Materialization;
	class Layout;
	class Buffer;
	class TensorView final
	{
	public:
		TensorView(const TensorSpec& spec, const Materialization& materialization);

		[[nodiscard]] const Shape& shape() const noexcept;
		[[nodiscard]] DType dtype() const noexcept;
		[[nodiscard]] Device device() const noexcept;
		[[nodiscard]] const Layout& layout() const noexcept;
		[[nodiscard]] const Buffer& buffer() const noexcept;

	private:
		const TensorSpec* spec_;
		const Materialization* materialization_;
	};

	class MutableTensorView final
	{
	public:
		MutableTensorView(const TensorSpec& spec, const Materialization& materialization);

		[[nodiscard]] const Shape& shape() const noexcept;
		[[nodiscard]] DType dtype() const noexcept;
		[[nodiscard]] Device device() const noexcept;
		[[nodiscard]] const Layout& layout() const noexcept;
		[[nodiscard]] Buffer& buffer() const noexcept;

	private:
		const TensorSpec* spec_;
		const Materialization* materialization_;
	};
}