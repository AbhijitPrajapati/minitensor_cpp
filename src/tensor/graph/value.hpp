#pragma once

#include <optional>

#include "ids.hpp"
#include "fwd.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
    class Node;
    class Value final
    {
    public:
        Value(ValueId id, TensorSpec spec);
        Value(ValueId id, TensorSpec spec, NodeRef producer);
        [[nodiscard]] ValueId id() const noexcept;
        [[nodiscard]] const TensorSpec &spec() const noexcept;
        [[nodiscard]] const Materialization *materialization() const noexcept;
        void materialize(Materialization materialization) const;
        [[nodiscard]] bool is_leaf() const noexcept;
        [[nodiscard]] const Node *producer() const noexcept;
        [[nodiscard]] const NodeRef &producer_ref() const noexcept;

    private:
        ValueId id_;
        TensorSpec spec_;
        NodeRef producer_;
        mutable std::optional<Materialization> materialization_;
        // Add EvaluationState here with the storage slice.
    };
}
