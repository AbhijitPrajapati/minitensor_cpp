#include <minitensor/ops.hpp>

#include <array>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/primitives/add.hpp"
#include "tensor/primitives/broadcast_to.hpp"
#include "tensor/primitives/divide.hpp"
#include "tensor/primitives/full.hpp"
#include "tensor/primitives/matmul.hpp"
#include "tensor/primitives/multiply.hpp"
#include "tensor/primitives/negate.hpp"
#include "tensor/primitives/permute.hpp"
#include "tensor/primitives/reshape.hpp"
#include "tensor/primitives/subtract.hpp"
#include "tensor/primitives/sum.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
    namespace
    {
        template <typename... Inputs>
        Tensor apply_primitive(std::unique_ptr<detail::Primitive> primitive, const Inputs &...inputs)
        {
            std::array<detail::ValueRef, sizeof...(Inputs)> input_values{
                detail::TensorAccess::value(inputs)...};
            detail::ValueRef output = detail::apply_operation(std::move(primitive), input_values);
            return detail::TensorAccess::make(std::move(output));
        }
    }

    Tensor operator+(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_primitive(std::make_unique<detail::AddPrimitive>(), lhs, rhs);
    }

    Tensor operator-(const Tensor &input)
    {
        return apply_primitive(std::make_unique<detail::NegatePrimitive>(), input);
    }

    Tensor operator-(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_primitive(std::make_unique<detail::SubtractPrimitive>(), lhs, rhs);
    }

    Tensor operator*(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_primitive(std::make_unique<detail::MultiplyPrimitive>(), lhs, rhs);
    }

    Tensor operator/(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_primitive(std::make_unique<detail::DividePrimitive>(), lhs, rhs);
    }

    Tensor full(Shape shape, float value, TensorOptions options)
    {
        detail::TensorSpec output_spec{std::move(shape), options.dtype, options.device};
        return apply_primitive(std::make_unique<detail::FullPrimitive>(std::move(output_spec), value));
    }

    Tensor permute(const Tensor &input, std::span<const Axis> permutation)
    {
        return apply_primitive(
            std::make_unique<detail::PermutePrimitive>(permutation, input.rank()), input);
    }

    Tensor reshape(const Tensor &input, Shape shape)
    {
        return apply_primitive(
            std::make_unique<detail::ReshapePrimitive>(std::move(shape)), input);
    }

    Tensor broadcast_to(const Tensor &input, Shape shape)
    {
        return apply_primitive(
            std::make_unique<detail::BroadcastToPrimitive>(std::move(shape)), input);
    }

    Tensor sum(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        return apply_primitive(
            std::make_unique<detail::SumPrimitive>(axes, input.rank(), keep_dim), input);
    }

    Tensor sum(const Tensor &input, bool keep_dim)
    {
        std::vector<Axis> axes(input.rank());
        std::iota(axes.begin(), axes.end(), Axis{0});
        return sum(input, axes, keep_dim);
    }

    Tensor matmul(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_primitive(std::make_unique<detail::MatmulPrimitive>(), lhs, rhs);
    }
}
