#pragma once

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/buffer.hpp"

namespace minitensor::detail
{

    class TensorView final
    {
    public:
        TensorView(const TensorSpec &spec, const Materialization &materialization);

        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] DType dtype() const noexcept;
        [[nodiscard]] Device device() const noexcept;
        [[nodiscard]] const Layout &layout() const noexcept;
        [[nodiscard]] const Buffer &buffer() const noexcept;

    private:
        const TensorSpec *spec_;
        const Materialization *materialization_;
    };

    class MutableTensorView final
    {
    public:
        MutableTensorView(const TensorSpec &spec, const Materialization &materialization);

        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] DType dtype() const noexcept;
        [[nodiscard]] Device device() const noexcept;
        [[nodiscard]] const Layout &layout() const noexcept;
        [[nodiscard]] Buffer &buffer() const noexcept;

    private:
        const TensorSpec *spec_;
        const Materialization *materialization_;
    };
}