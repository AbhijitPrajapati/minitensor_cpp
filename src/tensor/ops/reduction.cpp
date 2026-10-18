#include <minitensor/ops/reduction.hpp>

#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/apply_operation.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor
{
Tensor sum(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
{
    if (axes.empty())
    {
        return input;
    }
    return detail::apply_operation(std::make_unique<detail::SumPrimitive>(axes, input.rank(), keep_dim), input);
}

Tensor sum(const Tensor &input, std::initializer_list<Axis> axes, bool keep_dim)
{
    return sum(input, std::span<const Axis>(axes.begin(), axes.size()), keep_dim);
}

Tensor sum(const Tensor &input, bool keep_dim)
{
    std::vector<Axis> axes(input.rank());
    std::iota(axes.begin(), axes.end(), Axis{0});
    return sum(input, axes, keep_dim);
}

Tensor mean(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
{
    if (axes.empty())
    {
        return input;
    }

    const Shape &input_shape = input.shape();
    auto primitive = std::make_unique<detail::SumPrimitive>(axes, input_shape.rank(), keep_dim);

    // Collect the reduced extents to compute the reduction size.
    const auto &reduced_axes = primitive->axes();
    std::vector<Extent> reduced_extents;
    reduced_extents.reserve(reduced_axes.size());
    for (const Shape::size_type axis : reduced_axes)
    {
        reduced_extents.push_back(input_shape[axis]);
    }

    const Tensor total = detail::apply_operation(std::move(primitive), input);

    // Constructing a Shape preserves the library's checked element-count semantics.
    const Shape::size_type reduction_size = Shape{std::move(reduced_extents)}.numel();

    return total / static_cast<float>(reduction_size);
}

Tensor mean(const Tensor &input, std::initializer_list<Axis> axes, bool keep_dim)
{
    return mean(input, std::span<const Axis>(axes.begin(), axes.size()), keep_dim);
}

Tensor mean(const Tensor &input, bool keep_dim)
{
    std::vector<Axis> axes(input.rank());
    std::iota(axes.begin(), axes.end(), Axis{0});
    return mean(input, axes, keep_dim);
}

Tensor logsumexp(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
{
    if (axes.empty())
    {
        return input;
    }
    return detail::apply_operation(std::make_unique<detail::LogSumExpPrimitive>(axes, input.rank(), keep_dim), input);
}

Tensor logsumexp(const Tensor &input, std::initializer_list<Axis> axes, bool keep_dim)
{
    return logsumexp(input, std::span<const Axis>(axes.begin(), axes.size()), keep_dim);
}

Tensor logsumexp(const Tensor &input, bool keep_dim)
{
    std::vector<Axis> axes(input.rank());
    std::iota(axes.begin(), axes.end(), Axis{0});
    return logsumexp(input, axes, keep_dim);
}

} // namespace minitensor
