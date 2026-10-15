#pragma once

#include "buffer.hpp"
#include "layout.hpp"

namespace minitensor::detail
{
	struct TensorSpec;
	
	// This class represents the materialization of a value after it is evaluated.
	class Materialization final
	{
	public:
		Materialization(BufferRef buffer, Layout layout);
		[[nodiscard]] const BufferRef& buffer_ref() const noexcept;
		[[nodiscard]] const Layout& layout() const noexcept;
		void validate(const TensorSpec& spec) const;

	private:
		Layout layout_;
		BufferRef buffer_;
	};
}
