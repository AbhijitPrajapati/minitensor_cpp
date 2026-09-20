#include <minitensor/ops.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/primitives/add.hpp"
#include "tensor/primitives/broadcast_to.hpp"
#include "tensor/primitives/divide.hpp"
#include "tensor/primitives/exponential.hpp"
#include "tensor/primitives/full.hpp"
#include "tensor/primitives/hyperbolic_tangent.hpp"
#include "tensor/primitives/logarithm.hpp"
#include "tensor/primitives/matmul.hpp"
#include "tensor/primitives/max.hpp"
#include "tensor/primitives/mean.hpp"
#include "tensor/primitives/min.hpp"
#include "tensor/primitives/multiply.hpp"
#include "tensor/primitives/negate.hpp"
#include "tensor/primitives/permute.hpp"
#include "tensor/primitives/reshape.hpp"
#include "tensor/primitives/square_root.hpp"
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

        [[nodiscard]] TensorOptions options_like(const Tensor &input) noexcept
        {
            return TensorOptions{input.dtype(), input.device()};
        }

        [[nodiscard]] std::vector<Axis> all_axes(const Tensor &input)
        {
            std::vector<Axis> axes(input.rank());
            std::iota(axes.begin(), axes.end(), Axis{0});
            return axes;
        }

        [[nodiscard]] Extent flattened_extent(
            const Shape &shape,
            Shape::size_type start_axis,
            Shape::size_type end_axis)
        {
            for (Shape::size_type axis = start_axis; axis <= end_axis; ++axis)
            {
                if (shape[axis] == 0)
                {
                    return 0;
                }
            }

            Extent extent = 1;
            constexpr Extent max_extent = std::numeric_limits<Extent>::max();
            for (Shape::size_type axis = start_axis; axis <= end_axis; ++axis)
            {
                if (extent > max_extent / shape[axis])
                {
                    throw std::overflow_error{"flattened extent overflow"};
                }
                extent *= shape[axis];
            }
            return extent;
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

    Tensor exp(const Tensor &input)
    {
        return apply_primitive(std::make_unique<detail::ExponentialPrimitive>(), input);
    }

    Tensor log(const Tensor &input)
    {
        return apply_primitive(std::make_unique<detail::LogarithmPrimitive>(), input);
    }

    Tensor sqrt(const Tensor &input)
    {
        return apply_primitive(std::make_unique<detail::SquareRootPrimitive>(), input);
    }

    Tensor tanh(const Tensor &input)
    {
        return apply_primitive(
            std::make_unique<detail::HyperbolicTangentPrimitive>(), input);
    }

    Tensor full(Shape shape, float value, TensorOptions options)
    {
        detail::TensorSpec output_spec{std::move(shape), options.dtype, options.device};
        return apply_primitive(std::make_unique<detail::FullPrimitive>(std::move(output_spec), value));
    }

    Tensor full_like(const Tensor &input, float value)
    {
        return full(input.shape(), value, options_like(input));
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

    Tensor permute(const Tensor &input, std::span<const Axis> permutation)
    {
        return apply_primitive(
            std::make_unique<detail::PermutePrimitive>(permutation, input.rank()), input);
    }

    Tensor transpose(const Tensor &input)
    {
        if (input.rank() < 2)
        {
            return input;
        }

        std::vector<Axis> permutation(input.rank());
        std::iota(permutation.begin(), permutation.end(), Axis{0});
        std::reverse(permutation.begin(), permutation.end());
        return permute(input, permutation);
    }

    Tensor transpose(const Tensor &input, Axis axis0, Axis axis1)
    {
        const Shape::size_type normalized_axis0 = detail::normalize_axis(axis0, input.rank());
        const Shape::size_type normalized_axis1 = detail::normalize_axis(axis1, input.rank());
        if (normalized_axis0 == normalized_axis1)
        {
            return input;
        }

        std::vector<Axis> permutation(input.rank());
        std::iota(permutation.begin(), permutation.end(), Axis{0});
        std::swap(permutation[normalized_axis0], permutation[normalized_axis1]);
        return permute(input, permutation);
    }

    Tensor reshape(const Tensor &input, Shape shape)
    {
        return apply_primitive(
            std::make_unique<detail::ReshapePrimitive>(std::move(shape)), input);
    }

    Tensor flatten(const Tensor &input)
    {
        if (input.shape().is_scalar())
        {
            return reshape(input, Shape{1});
        }
        return flatten(input, 0, -1);
    }

    Tensor flatten(const Tensor &input, Axis start_axis, Axis end_axis)
    {
        const Shape::size_type normalized_start = detail::normalize_axis(start_axis, input.rank());
        const Shape::size_type normalized_end = detail::normalize_axis(end_axis, input.rank());
        if (normalized_start > normalized_end)
        {
            throw std::invalid_argument{"flatten start axis must not follow the end axis"};
        }
        if (normalized_start == normalized_end)
        {
            return input;
        }
        const auto dimensions = input.shape().dimensions();
        std::vector<Extent> output_dimensions;
        output_dimensions.reserve(input.rank() - (normalized_end - normalized_start));
        output_dimensions.insert(output_dimensions.end(), dimensions.begin(), dimensions.begin() + normalized_start);
        output_dimensions.push_back(flattened_extent(input.shape(), normalized_start, normalized_end));
        output_dimensions.insert(output_dimensions.end(), dimensions.begin() + normalized_end + 1, dimensions.end());
        return reshape(input, Shape{std::move(output_dimensions)});
    }

    Tensor squeeze(const Tensor &input)
    {
        std::vector<Extent> output_dimensions;
        output_dimensions.reserve(input.rank());
        for (const Extent extent : input.shape().dimensions())
        {
            if (extent != 1)
            {
                output_dimensions.push_back(extent);
            }
        }

        Shape output_shape{std::move(output_dimensions)};
        if (output_shape == input.shape())
        {
            return input;
        }
        return reshape(input, std::move(output_shape));
    }

    Tensor squeeze(const Tensor &input, Axis axis)
    {
        const std::array<Axis, 1> axes{axis};
        return squeeze(input, axes);
    }

    Tensor squeeze(const Tensor &input, std::span<const Axis> axes)
    {
        if (axes.empty())
        {
            return input;
        }

        std::vector<bool> removed_axes(input.rank(), false);
        for (const Axis axis : axes)
        {
            const Shape::size_type normalized = detail::normalize_axis(axis, input.rank());
            if (removed_axes[normalized])
            {
                throw std::invalid_argument{"squeeze axes must not contain duplicates"};
            }
            if (input.shape()[normalized] != 1)
            {
                throw std::invalid_argument{"squeeze requires every selected axis to have extent 1"};
            }
            removed_axes[normalized] = true;
        }

        std::vector<Extent> output_dimensions;
        output_dimensions.reserve(input.rank() - axes.size());
        for (Shape::size_type axis = 0; axis < input.rank(); ++axis)
        {
            if (!removed_axes[axis])
            {
                output_dimensions.push_back(input.shape()[axis]);
            }
        }
        return reshape(input, Shape{std::move(output_dimensions)});
    }

    Tensor unsqueeze(const Tensor &input, Axis axis)
    {
        const Shape::size_type output_rank = input.rank() + 1;
        const Shape::size_type normalized = detail::normalize_axis(axis, output_rank);
        const auto dimensions = input.shape().dimensions();

        std::vector<Extent> output_dimensions;
        output_dimensions.reserve(output_rank);
        output_dimensions.insert(output_dimensions.end(), dimensions.begin(), dimensions.begin() + normalized);
        output_dimensions.push_back(1);
        output_dimensions.insert(output_dimensions.end(), dimensions.begin() + normalized, dimensions.end());
        return reshape(input, Shape{std::move(output_dimensions)});
    }

    Tensor broadcast_to(const Tensor &input, Shape shape)
    {
        return apply_primitive(
            std::make_unique<detail::BroadcastToPrimitive>(std::move(shape)), input);
    }

    Tensor sum(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        if (axes.empty())
        {
            return input;
        }
        return apply_primitive(
            std::make_unique<detail::SumPrimitive>(axes, input.rank(), keep_dim), input);
    }

    Tensor sum(const Tensor &input, bool keep_dim)
    {
        return sum(input, all_axes(input), keep_dim);
    }

    Tensor mean(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        if (axes.empty())
        {
            return input;
        }
        return apply_primitive(
            std::make_unique<detail::MeanPrimitive>(axes, input.rank(), keep_dim), input);
    }

    Tensor mean(const Tensor &input, bool keep_dim)
    {
        return mean(input, all_axes(input), keep_dim);
    }

    Tensor max(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        if (axes.empty())
        {
            return input;
        }
        return apply_primitive(
            std::make_unique<detail::MaxPrimitive>(axes, input.rank(), keep_dim), input);
    }

    Tensor max(const Tensor &input, bool keep_dim)
    {
        return max(input, all_axes(input), keep_dim);
    }

    Tensor min(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        if (axes.empty())
        {
            return input;
        }
        return apply_primitive(
            std::make_unique<detail::MinPrimitive>(axes, input.rank(), keep_dim), input);
    }

    Tensor min(const Tensor &input, bool keep_dim)
    {
        return min(input, all_axes(input), keep_dim);
    }

    Tensor matmul(const Tensor &lhs, const Tensor &rhs)
    {
        return apply_primitive(std::make_unique<detail::MatmulPrimitive>(), lhs, rhs);
    }
}
