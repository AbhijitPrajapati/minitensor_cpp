#include "node.hpp"

#include <stdexcept>
#include <utility>

namespace minitensor::detail
{
    Node::Node(std::unique_ptr<Primitive> primitive, std::vector<ValueRef> inputs) : primitive_(std::move(primitive)), inputs_(std::move(inputs))
    {
        if (!primitive_)
        {
            throw std::invalid_argument{"Node requires a primitive"};
        }
        for (const ValueRef &input : inputs_)
        {
            if (!input)
            {
                throw std::invalid_argument{"Node inputs cannot be null"};
            }
        }
    }

    const Primitive &Node::primitive() const noexcept
    {
        return *primitive_;
    }

    std::span<const ValueRef> Node::inputs() const noexcept
    {
        return inputs_;
    }
}
