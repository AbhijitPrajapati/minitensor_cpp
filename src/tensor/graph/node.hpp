#pragma once

#include <span>
#include <vector>
#include <memory>

#include "primitive.hpp"
#include "fwd.hpp"

namespace minitensor::detail
{
    class Node final
    {
    public:
        Node(std::unique_ptr<Primitive> primitive, std::vector<ValueRef> inputs);
        [[nodiscard]] const Primitive &primitive() const noexcept;
        [[nodiscard]] std::span<const ValueRef> inputs() const noexcept;

    private:
        std::unique_ptr<Primitive> primitive_;
        std::vector<ValueRef> inputs_;
    };
}
