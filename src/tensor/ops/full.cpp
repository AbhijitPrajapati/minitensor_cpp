#include "full.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include <minitensor/ops.hpp>

#include "tensor/graph/apply_operation.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
    namespace detail
    {
        FullPrimitive::FullPrimitive(TensorSpec output_spec, float fill_value) : output_spec_(output_spec), fill_value_(fill_value) {}

        std::string_view FullPrimitive::name() const noexcept
        {
            return "full";
        }

        TensorSpec FullPrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (!inputs.empty())
            {
                throw std::invalid_argument{"full expects no input tensors"};
            }
            return output_spec_;
        }

        float FullPrimitive::fill_value() const noexcept
        {
            return fill_value_;
        }
    }

    Tensor full(Shape shape, float value, TensorOptions options)
    {
        detail::TensorSpec output_spec{std::move(shape), options.dtype, options.device};
        auto primitive = std::make_shared<detail::FullPrimitive>(std::move(output_spec), value);
        detail::ValueRef output = detail::apply_operation(std::move(primitive), {});
        return detail::TensorAccess::make(std::move(output));
    }
}
