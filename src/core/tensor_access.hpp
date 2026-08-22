#pragma once

#include "core/layout.hpp"
#include "minitensor/tensor.hpp"

#include <memory>
#include <span>

namespace minitensor::detail {

struct Node;

struct ConstTensorView final {
    std::span<const float> storage;
    const Layout& layout;
};

struct MutableTensorView final {
    std::span<float> storage;
    const Layout& layout;
};

struct TensorAccess final {
    using Identity = const void*;

    // Kernel orchestration receives only the layout and bounded storage spans,
    // never the complete TensorImpl.
    [[nodiscard]] static ConstTensorView view(const Tensor& tensor) noexcept;
    [[nodiscard]] static MutableTensorView mutable_view(Tensor& tensor) noexcept;

    // Autograd needs stable identity and graph-specific mutations, not general
    // access to storage or layout internals.
    [[nodiscard]] static Identity identity(const Tensor& tensor) noexcept;
    [[nodiscard]] static const Node* grad_fn(const Tensor& tensor) noexcept;
    static void set_grad_fn(Tensor& tensor, std::unique_ptr<Node> node) noexcept;
    static void set_grad(Tensor& tensor, const Tensor& gradient) noexcept;
};

} // namespace minitensor::detail
