#pragma once

#include "tensor/graph/primitive.hpp"
#include "tensor/core/tensor_spec.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace minitensor::detail
{
    class AddPrimitive final : public Primitive
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
    };

}