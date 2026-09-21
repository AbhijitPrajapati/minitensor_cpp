#pragma once

#include <minitensor/tensor.hpp>

namespace minitensor
{
    [[nodiscard]] Tensor matmul(const Tensor &lhs, const Tensor &rhs);
}
