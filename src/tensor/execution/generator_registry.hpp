#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <minitensor/types.hpp>

#include "tensor/core/random.hpp"
#include "device_hash.hpp"

namespace minitensor::detail
{
    class GeneratorRegistry final
    {
    public:
        GeneratorRegistry();
        explicit GeneratorRegistry(std::uint64_t seed);
        GeneratorRegistry(const GeneratorRegistry &) = delete;
        GeneratorRegistry &operator=(const GeneratorRegistry &) = delete;

        void manual_seed(std::uint64_t seed);
        [[nodiscard]] RandomKey reserve_key(const Device &device);

    private:
        std::uint64_t seed_;
        std::unordered_map<Device, std::unique_ptr<Generator>, DeviceHash> generators_;
        std::mutex mutex_;
    };
}
