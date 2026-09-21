#pragma once

#include <span>
#include <string_view>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class FullPrimitive final : public Primitive
    {
    public:
        explicit FullPrimitive(TensorSpec output_spec, float fill_value);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] float fill_value() const noexcept;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &) const override;

    private:
        TensorSpec output_spec_;
        float fill_value_;
    };
}
