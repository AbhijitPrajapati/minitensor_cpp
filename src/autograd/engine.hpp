#pragma once

#include "minitensor/tensor.hpp"

namespace minitensor::detail::autograd
{

    void run_backward(const Tensor &root, const Tensor &gradient);

} // namespace minitensor::detail::autograd
