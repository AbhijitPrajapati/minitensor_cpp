#pragma once

#include "autograd/recording.hpp"

#include <string>
#include <vector>

namespace minitensor::detail::autograd
{
    struct Node final
    {
        std::string name;
        std::vector<Tensor> parents;
        std::vector<Tensor> saved_tensors;
        BackwardFn backward;
    };

} // namespace minitensor::detail::autograd
