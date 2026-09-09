#pragma once

#include <vector>
#include <span>
#include <string_view>

#include <minitensor/types.hpp>

#include "tensor/graph/view_primitive.hpp"

namespace minitensor::detail
{
    class PermutePrimitive final : public ViewPrimitive
    {
    public:
        PermutePrimitive(std::span<const Axis> permutation, Shape::size_type input_rank);
        std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] Layout derive_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &permutation() const noexcept;

    private:
        std::vector<Shape::size_type> permutation_;
    };
}