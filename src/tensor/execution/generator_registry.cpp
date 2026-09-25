#include "generator_registry.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <random>

namespace minitensor::detail
{
    namespace
    {
        std::uint64_t random_seed()
        {
            std::random_device source;
            const auto high = static_cast<std::uint64_t>(source());
            const auto low = static_cast<std::uint64_t>(source());
            return (high << 32) ^ low;
        }
    }

    GeneratorRegistry::GeneratorRegistry()
        : GeneratorRegistry{random_seed()}
    {
    }

    GeneratorRegistry::GeneratorRegistry(std::uint64_t seed)
        : seed_{seed}
    {
    }

    void GeneratorRegistry::manual_seed(std::uint64_t seed)
    {
        const std::lock_guard lock{mutex_};
        seed_ = seed;
        for (auto &[device, generator] : generators_)
        {
            (void)device;
            generator->manual_seed(seed);
        }
    }

    RandomKey GeneratorRegistry::reserve_key(const Device &device)
    {
        const std::lock_guard lock{mutex_};
        auto [iterator, inserted] = generators_.try_emplace(device);
        if (inserted)
        {
            iterator->second = std::make_unique<Generator>(seed_);
        }
        return iterator->second->reserve_key();
    }
}
