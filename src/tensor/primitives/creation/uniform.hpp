#pragma once

#include <span>
#include <string_view>

#include "tensor/core/random.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class UniformParameters final
    {
    public:
        UniformParameters(float low, float high);
        [[nodiscard]] float low() const noexcept;
        [[nodiscard]] float high() const noexcept;

    private:
        float low_;
        float high_;
    };

    class UniformPrimitive final : public Primitive
    {
    public:
        explicit UniformPrimitive(TensorSpec output_spec, UniformParameters parameters, RandomKey key);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] float low() const noexcept;
        [[nodiscard]] float high() const noexcept;
        [[nodiscard]] const RandomKey &key() const noexcept;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &) const override;

    private:
        TensorSpec output_spec_;
        UniformParameters parameters_;
        RandomKey key_;
    };
}
