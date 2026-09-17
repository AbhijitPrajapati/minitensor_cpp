#pragma once

#include <vector>
#include <span>

#include <minitensor/tensor.hpp>

namespace minitensor
{
    [[nodiscard]] std::vector<Tensor> vjp(const Tensor &output, std::span<const Tensor> inputs, const Tensor &output_cotangent);
    [[nodiscard]] std::vector<Tensor> grad(const Tensor& output, std::span<const Tensor> inputs);
}