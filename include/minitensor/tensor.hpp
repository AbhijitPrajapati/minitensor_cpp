#pragma once

#include "minitensor/types.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace minitensor
{

    namespace detail
    {

        // Friend struct used by operators to interface with TensorImpl
        struct TensorAccess;

        // Internal tensor implementation
        struct TensorImpl;

    } // namespace detail

    class Tensor
    {
    public:
        // Equivalent to zeros
        // Named factories below are preferred for explicit initialization
        explicit Tensor(Shape shape, bool requires_grad = false);

        static Tensor zeros(Shape shape, bool requires_grad = false);
        static Tensor ones(Shape shape, bool requires_grad = false);
        static Tensor from_data(
            std::vector<float> values,
            Shape shape,
            bool requires_grad = false);

        Tensor(const Tensor &) noexcept = default;
        Tensor(Tensor &&) noexcept = default;
        Tensor &operator=(const Tensor &) noexcept = default;
        Tensor &operator=(Tensor &&) noexcept = default;
        ~Tensor();

        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] const Strides &strides() const noexcept;
        [[nodiscard]] Index storage_offset() const noexcept;
        [[nodiscard]] Index rank() const noexcept;
        [[nodiscard]] Index numel() const noexcept;
        [[nodiscard]] bool is_contiguous() const noexcept;
        [[nodiscard]] bool requires_grad() const noexcept;

        // Returns row-major data
        [[nodiscard]] std::vector<float> to_vector() const;
        [[nodiscard]] float item() const;

        [[nodiscard]] Tensor transpose(Index dim0, Index dim1) const;
        [[nodiscard]] Tensor slice(
            Index dim,
            Index start,
            Index stop,
            Index step = 1) const;
        [[nodiscard]] Tensor contiguous() const;
        [[nodiscard]] Tensor reshape(Shape shape) const;
        [[nodiscard]] Tensor detach() const;

        // Reduces every element if no dimension is provided
        // [[nodiscard]] Tensor sum(
        //     std::optional<Index> dim = std::nullopt,
        //     bool keepdim = false) const;

        [[nodiscard]] std::optional<Tensor> grad() const;
        void backward() const;
        void backward(const Tensor &gradient) const;
        void clear_grad();

    private:
        explicit Tensor(std::shared_ptr<detail::TensorImpl> impl) noexcept;

        // Shared pointer allows TensorImpl to be shared
        std::shared_ptr<detail::TensorImpl> impl_;

        // Friend declared so TensorAccess can access private impl_ property
        friend struct detail::TensorAccess;
    };

} // namespace minitensor
