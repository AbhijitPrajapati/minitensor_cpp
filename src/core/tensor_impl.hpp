#pragma once

#include "core/layout.hpp"
#include "core/storage.hpp"

#include <memory>

namespace minitensor::detail
{

    struct TensorImpl;

    namespace autograd
    {
        struct Node;

        struct Meta final
        {
            bool requires_grad{false};
            std::shared_ptr<TensorImpl> grad;
            std::unique_ptr<Node> grad_fn;

            explicit Meta(bool requires_grad = false);
            ~Meta();

            Meta(const Meta &) = delete;
            Meta &operator=(const Meta &) = delete;
        };
    } // namespace autograd

    struct TensorImpl final
    {
        std::shared_ptr<Storage> storage;
        Layout layout;
        autograd::Meta autograd;

        TensorImpl(std::shared_ptr<Storage> storage, Layout layout, bool requires_grad);
        ~TensorImpl();

        // Prevent copy construction and copy assignment
        // Cannot do TensorImpl t2 = t1;
        TensorImpl(const TensorImpl &) = delete;
        TensorImpl &operator=(const TensorImpl &) = delete;
    };

} // namespace minitensor::detail
