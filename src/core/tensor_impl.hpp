#pragma once

#include "core/layout.hpp"
#include "core/storage.hpp"

#include <memory>

namespace minitensor::detail
{

    struct Node;
    struct TensorImpl;

    struct AutogradMeta final
    {
        bool requires_grad{false};
        std::shared_ptr<TensorImpl> grad;
        std::unique_ptr<Node> grad_fn;

        explicit AutogradMeta(bool requires_grad = false);
        ~AutogradMeta();

        AutogradMeta(const AutogradMeta &) = delete;
        AutogradMeta &operator=(const AutogradMeta &) = delete;
    };

    struct TensorImpl final
    {
        std::shared_ptr<Storage> storage;
        Layout layout;
        AutogradMeta autograd;

        TensorImpl(std::shared_ptr<Storage> storage, Layout layout, bool requires_grad);
        ~TensorImpl();

        // Prevent copy construction and copy assignment
        // Cannot do TensorImpl t2 = t1;
        TensorImpl(const TensorImpl &) = delete;
        TensorImpl &operator=(const TensorImpl &) = delete;
    };

} // namespace minitensor::detail
