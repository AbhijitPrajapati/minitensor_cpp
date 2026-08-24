#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>
#include <vector>

namespace minitensor
{

    using Index = std::int64_t;

    class Shape final
    {
    public:
        using value_type = Index;
        using size_type = std::vector<Index>::size_type;
        using const_iterator = std::vector<Index>::const_iterator;
        using const_reverse_iterator = std::vector<Index>::const_reverse_iterator;

        // An empty shape represents a scalar and therefore contains one element.
        Shape() noexcept = default;
        Shape(std::initializer_list<Index> extents);
        Shape(std::vector<Index> extents);
        Shape(size_type rank, Index extent);

        [[nodiscard]] size_type size() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] Index rank() const noexcept;
        [[nodiscard]] Index numel() const noexcept;
        [[nodiscard]] Index operator[](size_type dimension) const noexcept;

        [[nodiscard]] const_iterator begin() const noexcept;
        [[nodiscard]] const_iterator end() const noexcept;
        [[nodiscard]] const_reverse_iterator rbegin() const noexcept;
        [[nodiscard]] const_reverse_iterator rend() const noexcept;

        void require_axis(Index dimension, std::string_view operation) const;
        void require_reshape_compatible(const Shape &other) const;
        [[nodiscard]] bool is_reshape_compatible_with(const Shape &other) const noexcept;

        [[nodiscard]] Shape broadcast_with(const Shape &other) const;
        [[nodiscard]] bool is_broadcastable_to(const Shape &target) const noexcept;
        [[nodiscard]] Shape reduced(std::optional<Index> dimension, bool keepdim) const;
        [[nodiscard]] Shape transposed(Index dim0, Index dim1) const;
        [[nodiscard]] Shape with_extent(Index dimension, Index extent) const;
        [[nodiscard]] Shape with_inserted_axis(Index dimension, Index extent = 1) const;

        [[nodiscard]] bool operator==(const Shape &other) const noexcept = default;

    private:
        std::vector<Index> extents_;
    };

    using Strides = std::vector<Index>;
    using Coordinates = std::vector<Index>;

} // namespace minitensor
