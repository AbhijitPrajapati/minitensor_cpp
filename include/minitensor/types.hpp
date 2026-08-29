#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

namespace minitensor
{
    // Shape, DType, DeviceType, Device, etc.
    using Extent = std::int64_t;
    using Axis = std::int64_t;

    class Shape final
    {
    public:
        using value_type = Extent;
        using size_type = std::size_t;

        // scalar shape
        Shape() = default;

        Shape(std::initializer_list<value_type> dimensions);
        explicit Shape(std::vector<value_type> dimensions);

        [[nodiscard]] size_type rank() const noexcept;
        [[nodiscard]] bool is_scalar() const noexcept;

        [[nodiscard]] value_type operator[](size_type axis) const noexcept;

        [[nodiscard]] std::span<const value_type> dimensions() const noexcept;

        [[nodiscard]] size_type numel() const noexcept;

        friend bool operator==(const Shape &, const Shape &) = default;

    private:
        [[nodiscard]] static size_type validate_and_count(std::span<const value_type> dimensions);

        std::vector<value_type> dimensions_;
        size_type numel_{1};
    };

    enum class DType : std::uint8_t
    {
        Float32,
    };

    [[nodiscard]] std::size_t dtype_size(DType dtype);
    [[nodiscard]] std::string_view dtype_name(DType dtype);

    enum class DeviceType : std::uint8_t
    {
        Cpu,
    };

    class Device final
    {
    public:
        constexpr Device() noexcept = default;
        [[nodiscard]]
        static constexpr Device cpu(std::uint32_t index = 0) noexcept
        {
            return Device{DeviceType::Cpu, index};
        }

        [[nodiscard]]
        constexpr DeviceType type() const noexcept
        {
            return type_;
        }

        [[nodiscard]]
        constexpr std::uint32_t index() const noexcept
        {
            return index_;
        }

        friend constexpr bool operator==(const Device &, const Device &) = default;

    private:
        constexpr Device(DeviceType type, std::uint32_t index) noexcept : type_{type}, index_{index} {}

        DeviceType type_{DeviceType::Cpu};
        std::uint32_t index_{0};
    };

    struct TensorOptions final
    {
        DType dtype{DType::Float32};
        Device device{Device::cpu()};

        friend bool operator==(const TensorOptions &, const TensorOptions &) = default;
    };

} // namespace minitensor
