#pragma once

#include <optional>

#include "fwd.hpp"
#include "node.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
	class Value final
	{
	public:
		Value(TensorSpec spec);
		Value(TensorSpec spec, NodeRef producer);
		[[nodiscard]] const TensorSpec& spec() const noexcept;
		[[nodiscard]] const Materialization* materialization() const noexcept;
		void materialize(Materialization materialization) const;
		[[nodiscard]] bool is_leaf() const noexcept;
		[[nodiscard]] const Node* producer() const noexcept;
		[[nodiscard]] const NodeRef& producer_ref() const noexcept;

	private:
		TensorSpec spec_;
		NodeRef producer_;
		mutable std::optional<Materialization> materialization_;
	};
}
