#pragma once
#include <string_view>
#include <vector>
#include <span>
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    class Primitive
    {
    public:
        virtual ~Primitive() = default;
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        [[nodiscard]] virtual TensorSpec infer(std::span<const TensorSpec> inputs) const = 0;
    };
}