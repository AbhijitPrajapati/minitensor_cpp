#pragma once

#include <span>

#include <minitensor/tensor.hpp>

namespace minitensor
{
    void eval(const Tensor &tensor);
    void eval(std::span<const Tensor> tensors);
}