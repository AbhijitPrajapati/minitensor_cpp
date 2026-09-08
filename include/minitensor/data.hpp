#pragma once

#include <span>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
    [[nodiscard]] std::vector<float> to_vector(const Tensor &tensor);
    [[nodiscard]] float item(const Tensor &tensor);
    [[nodiscard]] Tensor from_data(std::span<const float> data, Shape shape, TensorOptions options = {});
}