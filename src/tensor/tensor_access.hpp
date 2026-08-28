#pragma once

#include <minitensor/types.hpp>
#include <minitensor/tensor.hpp>
#include "graph/fwd.hpp"

namespace minitensor::detail
{
    struct TensorAccess final
    {
        [[nodiscard]] static const ValueRef &value(const Tensor &tensor) noexcept
        {
            return tensor.value_;
        }

        [[nodiscard]] static Tensor make(ValueRef value)
        {
            return Tensor{std::move(value)};
        }
    };
}