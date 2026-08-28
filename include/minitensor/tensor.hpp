#pragma once

#include "types.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace minitensor
{
    namespace detail
    {
        class Value;
        // friend struct allowing operators to access value
        struct TensorAccess;
    } // namespace detail

    class Tensor
    {
    public:
        Tensor(const Tensor &) noexcept = default;
        Tensor(Tensor &&) noexcept = default;
        Tensor &operator=(const Tensor &) noexcept = default;
        Tensor &operator=(Tensor &&) noexcept = default;

        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] std::size_t numel() const noexcept;
        [[nodiscard]] DType dtype() const noexcept;
        [[nodiscard]] Device device() const noexcept;

    private:
        explicit Tensor(std::shared_ptr<detail::Value>);
        std::shared_ptr<detail::Value> value_;
        // friend struct allowing operators to access internals
        friend struct detail::TensorAccess;
    };

} // namespace minitensor
