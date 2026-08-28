#pragma once

#include <memory>

namespace minitensor::detail
{
    class Value;
    class Node;
    class Primitive;

    using ValueRef = std::shared_ptr<Value>;
    using NodeRef = std::shared_ptr<const Node>;
    using PrimitiveRef = std::shared_ptr<const Primitive>;
}