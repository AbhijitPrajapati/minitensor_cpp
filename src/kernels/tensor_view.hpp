#pragma once

#include "core/layout.hpp"
#include "minitensor/tensor.hpp"

#include <span>

namespace minitensor::detail::kernel
{

    struct ConstTensorView final
    {
        std::span<const float> storage;
        const Layout &layout;
    };

    struct MutableTensorView final
    {
        std::span<float> storage;
        const Layout &layout;
    };

    struct TensorViewAccess final
    {
        [[nodiscard]] static ConstTensorView view(const Tensor &tensor) noexcept;
        [[nodiscard]] static MutableTensorView mutable_view(Tensor &tensor) noexcept;
    };

} // namespace minitensor::detail::kernel
