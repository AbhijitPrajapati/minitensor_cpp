#pragma once

#include <span>
#include <string_view>

#include <minitensor/random.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
class UniformPrimitive final : public Primitive
{
  public:
    static constexpr PrimitiveId ID{0x0002};
    [[nodiscard]] PrimitiveId id() const noexcept override
    {
        return ID;
    }
    explicit UniformPrimitive(TensorSpec output_spec, float low, float high, RandomKey key);
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
    [[nodiscard]] float low() const noexcept;
    [[nodiscard]] float high() const noexcept;
    [[nodiscard]] const RandomKey &key() const noexcept;

  private:
    TensorSpec output_spec_;
    float low_;
    float high_;
    RandomKey key_;
};
} // namespace minitensor::detail
