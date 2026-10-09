#include "leaf.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include <tensor/core/tensor_spec.hpp>
#include <tensor/storage/materialization.hpp>

#include "fwd.hpp"
#include "value.hpp"

namespace minitensor::detail
{
	ValueRef make_materialized_leaf(TensorSpec spec, Materialization materialization)
	{
		auto value = std::make_shared<Value>(std::move(spec));
		value->materialize(std::move(materialization));
		return value;
	}

	ValueRef leafify_materialized(const ValueRef& value)
	{
		if (!value)
		{
			throw std::invalid_argument{ "cannot leafify null value" };
		}

		const Materialization* materialization = value->materialization();
		if (materialization == nullptr)
		{
			throw std::logic_error{ "cannot leafify an unmaterialized value" };
		}

		if (value->is_leaf())
		{
			return value;
		}

		return make_materialized_leaf(value->spec(), *materialization);
	}
}