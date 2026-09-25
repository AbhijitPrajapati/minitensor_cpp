#include "random.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>

namespace minitensor::detail
{
    Generator::Generator(std::uint64_t seed)
        : seed_{seed}, next_stream_{0}
    {
    }

    void Generator::manual_seed(std::uint64_t seed)
    {
        const std::lock_guard lock{mutex_};
        seed_ = seed;
        next_stream_ = 0;
        exhausted_ = false;
    }

    RandomKey Generator::reserve_key()
    {
        const std::lock_guard lock{mutex_};
        if (exhausted_)
        {
            throw std::overflow_error{"random generator stream exhausted"};
        }

        const RandomKey key{seed_, next_stream_};
        if (next_stream_ == std::numeric_limits<std::uint64_t>::max())
        {
            exhausted_ = true;
        }
        else
        {
            ++next_stream_;
        }
        return key;
    }
}
