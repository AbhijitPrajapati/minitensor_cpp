#pragma once
#include <cstdint>
#include <variant>
#include "fwd.hpp"

namespace minitensor::detail
{

    struct LeafOrigin final
    {
    };

    struct ProducedOrigin final
    {
        NodeRef node;
    };

    using Origin = std::variant<LeafOrigin, ProducedOrigin>;
}