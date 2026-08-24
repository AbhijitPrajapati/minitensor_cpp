#pragma once

#include "minitensor/tensor.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace minitensor::detail
{

    using GradList = std::vector<std::optional<Tensor>>;
    using BackwardFn = std::function<GradList(const Tensor &grad_output)>;

    struct Node final
    {
        std::string name;
        std::vector<Tensor> parents;
        BackwardFn backward;
    };

    void set_history(
        Tensor &result,
        std::string name,
        std::vector<Tensor> parents,
        BackwardFn backward);

} // namespace minitensor::detail
