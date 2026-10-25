#pragma once

#include <optional>
#include <span>
#include <string_view>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
class ReshapePrimitive final : public Primitive
{
  public:
    static constexpr PrimitiveId ID{0x0202};
    [[nodiscard]] PrimitiveId id() const noexcept override
    {
        return ID;
    }
    explicit ReshapePrimitive(Shape shape);
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
    [[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec &input_spec,
                                                                 const Layout &input_layout,
                                                                 const TensorSpec &output_spec) const override;
    [[nodiscard]] const Shape &shape() const noexcept;

  private:
    Shape shape_;
};
} // namespace minitensor::detail
