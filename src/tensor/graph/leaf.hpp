#pragma once

#include "fwd.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
	[[nodiscard]] ValueRef make_materialized_leaf(TensorSpec spec, Materialization materialization);
	[[nodiscard]] ValueRef leafify_materialized(const ValueRef& value);
}