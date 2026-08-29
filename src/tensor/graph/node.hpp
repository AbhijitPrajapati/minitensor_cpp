#pragma once

#include <span>
#include <vector>

#include "fwd.hpp"
#include "ids.hpp"

namespace minitensor::detail
{
    class Node final
    {
    public:
        Node(NodeId id, PrimitiveRef primitive, std::vector<ValueRef> inputs);

        [[nodiscard]] NodeId id() const noexcept;
        [[nodiscard]] const Primitive &primitive() const noexcept;
        [[nodiscard]] const PrimitiveRef &primitive_ref() const noexcept;

        [[nodiscard]] std::span<const ValueRef> inputs() const noexcept;

    private:
        NodeId id_;
        PrimitiveRef primitive_;
        std::vector<ValueRef> inputs_;
    };
}
