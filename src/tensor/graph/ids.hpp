#pragma once

#include <cstdint>

namespace minitensor::detail
{
    struct ValueId final
    {
        std::uint64_t value;
        friend bool operator==(ValueId, ValueId) = default;
    };

    struct NodeId final
    {
        std::uint64_t value;
        friend bool operator==(NodeId, NodeId) = default;
    };

    ValueId next_value_id() noexcept;
    NodeId next_node_id() noexcept;
}
