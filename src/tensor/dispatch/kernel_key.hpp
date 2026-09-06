#pragma once

#include <typeindex>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
    using PrimitiveTypeId = std::type_index;

    struct KernelKey final
    {
        PrimitiveTypeId primitive_type;
        DeviceType device_type;
        DType dtype;
        friend bool operator==(const KernelKey &, const KernelKey &) = default;
    };

    struct KernelKeyHash final {
        [[nodiscard]] std::size_t operator()(const KernelKey& key) const noexcept {
            std::size_t result = key.primitive_type.hash_code();
            combine(result, hash_enum(key.device_type));
            combine(result, hash_enum(key.dtype));
            return result;
        }

    private:
        static void combine(std::size_t& seed, std::size_t value) noexcept {
            seed ^= value + std::size_t{ 0x9e3779b9 } + (seed << 6) + (seed >> 2);
        }

        template <typename Enum>
        [[nodiscard]] static std::size_t hash_enum(Enum value) noexcept {
            using Underlying = std::underlying_type_t<Enum>;

            return std::hash<Underlying>{}(
                static_cast<Underlying>(value));
        }
    };
}