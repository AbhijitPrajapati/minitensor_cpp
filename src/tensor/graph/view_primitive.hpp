#pragma once

#include "primitive.hpp"

namespace minitensor::detail
{
    class Layout;

    class ViewPrimitive : public Primitive
    {
    public:
        ~ViewPrimitive() override = default;
        [[nodiscard]] virtual Layout derive_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const = 0;
    };
}