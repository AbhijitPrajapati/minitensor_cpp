#include "kernels/iteration.hpp"
#include "core/shape.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace minitensor::detail::kernel
{

    void advance_coordinates(const Shape &shape, Coordinates &coordinates) noexcept
    {
        for (auto dimension = shape.size(); dimension > 0; --dimension)
        {
            const auto axis = dimension - 1;
            ++coordinates[axis];
            if (coordinates[axis] < shape[axis])
            {
                break;
            }
            coordinates[axis] = 0;
        }
    }

    ElementwiseIterator::ElementwiseIterator(
        const Layout &output,
        const std::initializer_list<Layout> inputs)
        : shape_(output.shape()),
          numel_(output.numel())
    {
        layouts_.reserve(inputs.size() + 1);
        layouts_.push_back(output);

        if (inputs.size() == 0)
        {
            return;
        }

        for (const auto &input : inputs)
        {
            if (!shape::is_broadcastable_to(input.shape(), shape_))
            {
                throw std::logic_error(
                    "elementwise input shape cannot broadcast to the output shape");
            }
            layouts_.push_back(input.broadcast_to(shape_));
        }
    }

    const Shape &ElementwiseIterator::shape() const noexcept
    {
        return shape_;
    }

    Index ElementwiseIterator::numel() const noexcept
    {
        return numel_;
    }

    std::size_t ElementwiseIterator::input_count() const noexcept
    {
        return layouts_.size() - 1;
    }

    const Layout &ElementwiseIterator::output_layout() const noexcept
    {
        return layouts_.front();
    }

    const Layout &ElementwiseIterator::input_layout(const std::size_t index) const
    {
        if (index >= input_count())
        {
            throw std::out_of_range("elementwise input layout index is out of range");
        }
        return layouts_[index + 1];
    }

    ReductionIterator::ReductionIterator(
        const Layout &input,
        const Layout &output,
        const std::optional<Index> dimension,
        const bool keepdim)
        : ReductionIterator(
              input,
              output,
              configure_standard(input, output, dimension, keepdim))
    {
    }

    ReductionIterator ReductionIterator::to_shape(
        const Layout &input,
        const Layout &output)
    {
        return ReductionIterator(input, output, configure_to_shape(input, output));
    }

    const Shape &ReductionIterator::input_shape() const noexcept
    {
        return input_shape_;
    }

    const Shape &ReductionIterator::output_shape() const noexcept
    {
        return output_shape_;
    }

    Index ReductionIterator::numel() const noexcept
    {
        return numel_;
    }

    std::span<const Index> ReductionIterator::reduced_axes() const noexcept
    {
        return reduced_axes_;
    }

    bool ReductionIterator::keepdim() const noexcept
    {
        return keepdim_;
    }

    const Layout &ReductionIterator::input_layout() const noexcept
    {
        return input_layout_;
    }

    const Layout &ReductionIterator::output_layout() const noexcept
    {
        return output_layout_;
    }

    ReductionIterator::ReductionIterator(
        const Layout &input,
        const Layout &output,
        Configuration configuration)
        : input_shape_(input.shape()),
          output_shape_(output.shape()),
          numel_(input.numel()),
          reduced_axes_(std::move(configuration.reduced_axes)),
          keepdim_(configuration.keepdim),
          reduced_axis_mask_(input.shape().size(), false),
          input_layout_(input),
          output_layout_(output),
          normalized_output_layout_(std::move(configuration.normalized_output_layout))
    {
        for (const auto axis : reduced_axes_)
        {
            reduced_axis_mask_[static_cast<std::size_t>(axis)] = true;
        }
    }

    ReductionIterator::Configuration ReductionIterator::configure_standard(
        const Layout &input,
        const Layout &output,
        const std::optional<Index> dimension,
        const bool keepdim)
    {
        const auto expected_output_shape = shape::reduce(input.shape(), dimension, keepdim);
        if (output.shape() != expected_output_shape)
        {
            throw std::logic_error("reduction output shape does not match its dimensions");
        }

        std::vector<Index> reduced_axes;
        if (dimension.has_value())
        {
            reduced_axes.push_back(*dimension);
        }
        else
        {
            reduced_axes.reserve(input.shape().size());
            for (Index axis = 0; axis < input.rank(); ++axis)
            {
                reduced_axes.push_back(axis);
            }
        }

        if (keepdim)
        {
            return Configuration{std::move(reduced_axes), true, output};
        }

        Shape normalized_shape;
        Strides normalized_strides;
        normalized_shape.reserve(input.shape().size());
        normalized_strides.reserve(input.shape().size());
        std::size_t output_axis = 0;
        for (Index input_axis = 0; input_axis < input.rank(); ++input_axis)
        {
            if (std::ranges::binary_search(reduced_axes, input_axis))
            {
                normalized_shape.push_back(1);
                normalized_strides.push_back(0);
            }
            else
            {
                normalized_shape.push_back(output.shape()[output_axis]);
                normalized_strides.push_back(output.strides()[output_axis]);
                ++output_axis;
            }
        }

        return Configuration{
            std::move(reduced_axes),
            false,
            Layout(
                std::move(normalized_shape),
                std::move(normalized_strides),
                output.offset())};
    }

    ReductionIterator::Configuration ReductionIterator::configure_to_shape(
        const Layout &input,
        const Layout &output)
    {
        if (!shape::is_broadcastable_to(output.shape(), input.shape()))
        {
            throw std::invalid_argument("reduction target cannot broadcast to the input shape");
        }

        const auto rank_difference = input.shape().size() - output.shape().size();
        Shape normalized_shape(input.shape().size(), 1);
        Strides normalized_strides(input.shape().size(), 0);
        std::vector<Index> reduced_axes;
        reduced_axes.reserve(input.shape().size());

        for (std::size_t input_axis = 0; input_axis < rank_difference; ++input_axis)
        {
            reduced_axes.push_back(static_cast<Index>(input_axis));
        }
        for (std::size_t output_axis = 0; output_axis < output.shape().size(); ++output_axis)
        {
            const auto input_axis = rank_difference + output_axis;
            normalized_shape[input_axis] = output.shape()[output_axis];
            normalized_strides[input_axis] = output.strides()[output_axis];
            if (output.shape()[output_axis] != input.shape()[input_axis])
            {
                reduced_axes.push_back(static_cast<Index>(input_axis));
            }
        }

        return Configuration{
            std::move(reduced_axes),
            input.rank() == output.rank(),
            Layout(
                std::move(normalized_shape),
                std::move(normalized_strides),
                output.offset())};
    }

    MatrixMultiplyPlan::MatrixMultiplyPlan(
        const Layout &lhs,
        const Layout &rhs,
        const Layout &output)
        : lhs_layout_(lhs),
          rhs_layout_(rhs),
          output_layout_(output)
    {
        if (lhs.rank() != 2 || rhs.rank() != 2 || output.rank() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 layouts");
        }
        if (lhs.shape()[1] != rhs.shape()[0])
        {
            throw std::invalid_argument("matmul inner dimensions do not match");
        }
        if (output.shape() != Shape{lhs.shape()[0], rhs.shape()[1]})
        {
            throw std::logic_error("matmul output has the wrong shape");
        }

        rows_ = lhs.shape()[0];
        inner_size_ = lhs.shape()[1];
        columns_ = rhs.shape()[1];
    }

    Index MatrixMultiplyPlan::rows() const noexcept
    {
        return rows_;
    }

    Index MatrixMultiplyPlan::inner_size() const noexcept
    {
        return inner_size_;
    }

    Index MatrixMultiplyPlan::columns() const noexcept
    {
        return columns_;
    }

    const Layout &MatrixMultiplyPlan::lhs_layout() const noexcept
    {
        return lhs_layout_;
    }

    const Layout &MatrixMultiplyPlan::rhs_layout() const noexcept
    {
        return rhs_layout_;
    }

    const Layout &MatrixMultiplyPlan::output_layout() const noexcept
    {
        return output_layout_;
    }

} // namespace minitensor::detail::kernel
