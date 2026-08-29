#pragma once

#include <span>
#include <string_view>

namespace minitensor::detail
{
    struct TensorSpec;

    class Primitive
    {
    public:
        virtual ~Primitive() = default;
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        [[nodiscard]] virtual TensorSpec infer(std::span<const TensorSpec> inputs) const = 0;
    };
}
