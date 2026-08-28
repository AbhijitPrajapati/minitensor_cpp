#pragma once

#include "tensor/core/tensor_spec.hpp"
#include "ids.hpp"
#include "origin.hpp"

namespace minitensor::detail
{
    class Value final
    {
    public:
        Value(ValueId id, TensorSpec spec, Origin origin);

        [[nodiscard]] ValueId id() const noexcept;
        [[nodiscard]] const TensorSpec &spec() const noexcept;
        [[nodiscard]] const Origin &origin() const noexcept;

    private:
        ValueId id_;
        TensorSpec spec_;
        Origin origin_;
        // Add EvaluationState here with the storage slice.
    };
}