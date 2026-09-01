#include "value.hpp"
#include "fwd.hpp"

#include <stdexcept>
#include <utility>

namespace minitensor::detail
{
    Value::Value(ValueId id, TensorSpec spec) : id_(id), spec_(std::move(spec)) {}

    Value::Value(ValueId id, TensorSpec spec, NodeRef producer) : id_(id), spec_(std::move(spec)), producer_(std::move(producer))
    {
        if (!producer_)
        {
            throw std::invalid_argument{"a produced value require a producer node"};
        }
    }

    ValueId Value::id() const noexcept
    {
        return id_;
    }

    const TensorSpec &Value::spec() const noexcept
    {
        return spec_;
    }

    const Materialization *Value::materialization() const noexcept
    {
        return materialization_ ? (&*materialization_) : nullptr;
    }

    void Value::materialize(Materialization materialization) const
    {
        if (materialization_)
        {
            throw std::logic_error{"value has already been materialized"};
        }
        materialization.validate(spec_);
        materialization_.emplace(std::move(materialization));
    }

    bool Value::is_leaf() const noexcept
    {
        return !producer_;
    }

    const Node *Value::producer() const noexcept
    {
        return producer_.get();
    }

    const NodeRef &Value::producer_ref() const noexcept
    {
        return producer_;
    }
}
