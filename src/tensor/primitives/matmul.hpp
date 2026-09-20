#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class MatmulPrimitive final : public Primitive
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const override;
    };
}
