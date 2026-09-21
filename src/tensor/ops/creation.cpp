#include <minitensor/ops/creation.hpp>

#include <memory>
#include <utility>

#include "apply_primitive.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/primitives/creation/full.hpp"

namespace minitensor
{
    Tensor full(Shape shape, float value, TensorOptions options)
    {
        detail::TensorSpec output_spec{std::move(shape), options.dtype, options.device};
        return detail::apply_primitive(
            std::make_unique<detail::FullPrimitive>(std::move(output_spec), value));
    }

    Tensor full_like(const Tensor &input, float value)
    {
        const TensorOptions options_like = TensorOptions{ input.dtype(), input.device() };
        return full(input.shape(), value, options_like);
    }

    Tensor full_like(const Tensor &input, float value, TensorOptions options)
    {
        return full(input.shape(), value, options);
    }

    Tensor zeros(Shape shape, TensorOptions options)
    {
        return full(std::move(shape), 0.0F, options);
    }

    Tensor ones(Shape shape, TensorOptions options)
    {
        return full(std::move(shape), 1.0F, options);
    }

    Tensor zeros_like(const Tensor &input)
    {
        return full_like(input, 0.0F);
    }

    Tensor zeros_like(const Tensor &input, TensorOptions options)
    {
        return full_like(input, 0.0F, options);
    }

    Tensor ones_like(const Tensor &input)
    {
        return full_like(input, 1.0F);
    }

    Tensor ones_like(const Tensor &input, TensorOptions options)
    {
        return full_like(input, 1.0F, options);
    }
}
