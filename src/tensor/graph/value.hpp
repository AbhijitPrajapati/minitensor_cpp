#pragma once

#include <optional>

#include "ids.hpp"
#include "origin.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
    class Value final
    {
    public:
        Value(ValueId id, TensorSpec spec, Origin origin);

        [[nodiscard]] ValueId id() const noexcept;
        [[nodiscard]] const TensorSpec &spec() const noexcept;
        [[nodiscard]] const Origin &origin() const noexcept;
        [[nodiscard]] const Materialization *materialization() const noexcept;
        void materialize(Materialization materialization) const;

    private:
        ValueId id_;
        TensorSpec spec_;
        Origin origin_;
        mutable std::optional<Materialization> materialization_;
        // Add EvaluationState here with the storage slice.
    };
}
