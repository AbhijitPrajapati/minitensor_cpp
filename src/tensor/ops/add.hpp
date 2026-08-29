#pragma once

#include <span>
#include <string_view>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class AddPrimitive final : public Primitive
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
    };

}
