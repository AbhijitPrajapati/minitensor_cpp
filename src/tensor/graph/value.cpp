#include "tensor/core/tensor_spec.hpp"
#include "ids.hpp"
#include "origin.hpp"
#include "value.hpp"

namespace minitensor::detail
{
    Value::Value(ValueId id, TensorSpec spec, Origin origin) : id_(id), spec_(std::move(spec)), origin_(std::move(origin)) {}

    ValueId Value::id() const noexcept
    {
        return id_;
    }

    const TensorSpec &Value::spec() const noexcept
    {
        return spec_;
    }

    const Origin &Value::origin() const noexcept
    {
        return origin_;
    }
}