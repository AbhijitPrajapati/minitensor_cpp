#pragma once

#include <span>
#include <vector>
#include <memory>

#include "ids.hpp"
#include "primitive.hpp"
#include "fwd.hpp"

namespace minitensor::detail
{
    class Node final
    {
    public:
        Node(NodeId id, std::unique_ptr<Primitive> primitive, std::vector<ValueRef> inputs);

        [[nodiscard]] NodeId id() const noexcept;
        [[nodiscard]] const Primitive &primitive() const noexcept;
        [[nodiscard]] std::span<const ValueRef> inputs() const noexcept;

    private:
        NodeId id_;
        std::unique_ptr<Primitive> primitive_;
        std::vector<ValueRef> inputs_;
    };
}
