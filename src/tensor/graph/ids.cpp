#include "ids.hpp"

#include <atomic>
#include <cstdint>

namespace minitensor::detail
{
    ValueId next_value_id() noexcept
    {
        static std::atomic<std::uint64_t> next{0};
        return ValueId{next.fetch_add(1, std::memory_order_relaxed)};
    }

    NodeId next_node_id() noexcept
    {
        static std::atomic<std::uint64_t> next{0};
        return NodeId{next.fetch_add(1, std::memory_order_relaxed)};
    }
}
