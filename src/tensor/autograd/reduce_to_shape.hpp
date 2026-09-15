#pragma once

#include <minitensor/types.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::detail
{
    [[nodiscard]] Tensor reduce_to_shape(const Tensor &cotangent, const Shape &target_shape);
}