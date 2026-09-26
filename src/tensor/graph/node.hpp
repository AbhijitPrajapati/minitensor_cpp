#pragma once

#include <memory>
#include <span>
#include <vector>

#include "fwd.hpp"
#include "primitive.hpp"

namespace minitensor::detail
{
	class Node final
	{
	public:
		Node(std::unique_ptr<Primitive> primitive, std::vector<ValueRef> inputs);
		[[nodiscard]] const Primitive& primitive() const noexcept;
		[[nodiscard]] std::span<const ValueRef> inputs() const noexcept;

	private:
		std::unique_ptr<Primitive> primitive_;
		std::vector<ValueRef> inputs_;
	};
}
