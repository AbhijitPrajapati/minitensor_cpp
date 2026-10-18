#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
class PermutePrimitive final : public Primitive
{
  public:
    static constexpr PrimitiveId ID{0x0203};
    [[nodiscard]] PrimitiveId id() const noexcept override
    {
        return ID;
    }
    explicit PermutePrimitive(std::span<const Axis> permutation, Shape::size_type input_rank);
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] bool requires_kernel_support() const noexcept override;
    [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
    [[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec &input_spec,
                                                                 const Layout &input_layout,
                                                                 const TensorSpec &output_spec) const override;
    [[nodiscard]] const std::vector<Shape::size_type> &permutation() const noexcept;

  private:
    std::vector<Shape::size_type> permutation_;
    Shape::size_type input_rank_;
};
} // namespace minitensor::detail
