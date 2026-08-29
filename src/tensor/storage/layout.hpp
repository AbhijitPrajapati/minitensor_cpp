#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace minitensor
{
    class Shape;
}

namespace minitensor::detail
{
    class Layout final
    {
    public:
        using stride_type = std::int64_t;
        using offset_type = std::int64_t;
        using size_type = std::size_t;

        // scalar layout
        Layout() = default;
        Layout(std::vector<stride_type> strides, offset_type offset = 0);

        [[nodiscard]] static Layout contiguous(const Shape &shape);
        [[nodiscard]] size_type rank() const noexcept;
        [[nodiscard]] stride_type stride(size_type axis) const noexcept;
        [[nodiscard]] std::span<const stride_type> strides() const noexcept;
        [[nodiscard]] offset_type offset() const noexcept;
        [[nodiscard]] bool is_contiguous(const Shape &shape) const noexcept;

        friend bool operator==(const Layout &, const Layout &) = default;

    private:
        std::vector<stride_type> strides_;
        offset_type offset_{0};
    };
}
