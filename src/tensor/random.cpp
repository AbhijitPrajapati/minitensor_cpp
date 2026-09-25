#include <minitensor/random.hpp>

#include "tensor/execution/environment.hpp"

namespace minitensor
{
    void manual_seed(std::uint64_t seed)
    {
        detail::environment().manual_seed(seed);
    }
}
