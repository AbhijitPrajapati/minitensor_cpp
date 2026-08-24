#include "kernels/kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace minitensor::detail
{
    namespace
    {

        std::size_t as_size(const Index value)
        {
            if (value < 0)
            {
                throw std::logic_error("a validated tensor index became negative");
            }
            return static_cast<std::size_t>(value);
        }

        float read_at(const std::span<const float> storage, const Index offset)
        {
            const auto index = as_size(offset);
            if (index >= storage.size())
            {
                throw std::out_of_range("kernel read is outside tensor storage");
            }
            return storage[index];
        }

        float &write_at(const std::span<float> storage, const Index offset)
        {
            const auto index = as_size(offset);
            if (index >= storage.size())
            {
                throw std::out_of_range("kernel write is outside tensor storage");
            }
            return storage[index];
        }

        // Returns a value from a virtually left-padded shape or stride vector.
        Index get_padded(
            const std::vector<Index> &values,
            const Index index,
            const Index padding_value,
            const Index rank_difference)
        {
            return index < rank_difference
                       ? padding_value
                       : values[as_size(index - rank_difference)];
        }

        void require_same_shape(const Layout &lhs, const Layout &rhs, const char *operation)
        {
            if (lhs.shape() != rhs.shape())
            {
                throw std::invalid_argument(std::string(operation) + " requires equal logical shapes");
            }
        }

        float apply_binary(const float lhs, const float rhs, const BinaryKernel operation)
        {
            switch (operation)
            {
            case BinaryKernel::add:
                return lhs + rhs;
            case BinaryKernel::subtract:
                return lhs - rhs;
            case BinaryKernel::multiply:
                return lhs * rhs;
            case BinaryKernel::divide:
                return lhs / rhs;
            }
            throw std::logic_error("unknown binary kernel");
        }

        float apply_unary(const float value, const UnaryKernel operation)
        {
            switch (operation)
            {
            case UnaryKernel::negate:
                return -value;
            case UnaryKernel::relu:
                return std::max(0.0F, value);
            case UnaryKernel::sigmoid:
                if (value >= 0.0F)
                {
                    const auto z = std::exp(-value);
                    return 1.0F / (1.0F + z);
                }
                else
                {
                    const auto z = std::exp(value);
                    return z / (1.0F + z);
                }
            case UnaryKernel::tanh:
                return std::tanh(value);
            }
            throw std::logic_error("unknown unary kernel");
        }

    } // namespace

    BroadcastPlan make_broadcast_plan(const Layout &lhs, const Layout &rhs)
    {
        const Index output_rank = std::max(lhs.rank(), rhs.rank());
        BroadcastPlan plan{
            Shape(as_size(output_rank), 1),
            Strides(as_size(output_rank), 0),
            Strides(as_size(output_rank), 0)};

        const Index lhs_difference = output_rank - lhs.rank();
        const Index rhs_difference = output_rank - rhs.rank();

        for (const Index dim : std::views::iota(Index{0}, output_rank))
        {
            const Index lhs_extent = get_padded(lhs.shape(), dim, 1, lhs_difference);
            const Index rhs_extent = get_padded(rhs.shape(), dim, 1, rhs_difference);
            if (lhs_extent != rhs_extent && lhs_extent != 1 && rhs_extent != 1)
            {
                throw std::invalid_argument("tensor shapes are not broadcast-compatible");
            }

            const Index output_extent = lhs_extent == 1 ? rhs_extent : lhs_extent;
            const auto index = as_size(dim);
            plan.output_shape[index] = output_extent;
            plan.lhs_strides[index] = lhs_extent == output_extent
                                          ? get_padded(lhs.strides(), dim, 0, lhs_difference)
                                          : 0;
            plan.rhs_strides[index] = rhs_extent == output_extent
                                          ? get_padded(rhs.strides(), dim, 0, rhs_difference)
                                          : 0;
        }
        return plan;
    }

    void fill(const WriteTensorArg output, const float value)
    {
        for (Index linear = 0; linear < output.layout.numel(); ++linear)
        {
            const auto coordinates = output.layout.coordinates_from_linear(linear);
            write_at(output.storage, output.layout.offset_from_coordinates(coordinates)) = value;
        }
    }

    void copy(const ReadTensorArg input, const WriteTensorArg output)
    {
        require_same_shape(input.layout, output.layout, "copy");
        for (Index linear = 0; linear < input.layout.numel(); ++linear)
        {
            const auto coordinates = input.layout.coordinates_from_linear(linear);
            const auto input_offset = input.layout.offset_from_coordinates(coordinates);
            const auto output_offset = output.layout.offset_from_coordinates(coordinates);
            write_at(output.storage, output_offset) = read_at(input.storage, input_offset);
        }
    }

    void binary(
        const ReadTensorArg lhs,
        const ReadTensorArg rhs,
        const WriteTensorArg output,
        const BroadcastPlan &plan,
        const BinaryKernel operation)
    {
        if (output.layout.shape() != plan.output_shape)
        {
            throw std::logic_error("binary kernel output shape does not match broadcast plan");
        }

        const Layout broadcast_lhs(plan.output_shape, plan.lhs_strides, lhs.layout.offset());
        const Layout broadcast_rhs(plan.output_shape, plan.rhs_strides, rhs.layout.offset());
        for (Index linear = 0; linear < output.layout.numel(); ++linear)
        {
            const auto coordinates = output.layout.coordinates_from_linear(linear);
            const auto lhs_offset = broadcast_lhs.offset_from_coordinates(coordinates);
            const auto rhs_offset = broadcast_rhs.offset_from_coordinates(coordinates);
            const auto output_offset = output.layout.offset_from_coordinates(coordinates);
            write_at(output.storage, output_offset) = apply_binary(
                read_at(lhs.storage, lhs_offset), read_at(rhs.storage, rhs_offset), operation);
        }
    }

    void unary(
        const ReadTensorArg input,
        const WriteTensorArg output,
        const UnaryKernel operation)
    {
        require_same_shape(input.layout, output.layout, "unary kernel");
        for (Index linear = 0; linear < input.layout.numel(); ++linear)
        {
            const auto coordinates = input.layout.coordinates_from_linear(linear);
            const auto input_offset = input.layout.offset_from_coordinates(coordinates);
            const auto output_offset = output.layout.offset_from_coordinates(coordinates);
            write_at(output.storage, output_offset) = apply_unary(
                read_at(input.storage, input_offset), operation);
        }
    }

    void matrix_multiply(
        const ReadTensorArg lhs,
        const ReadTensorArg rhs,
        const WriteTensorArg output)
    {
        if (lhs.layout.rank() != 2 || rhs.layout.rank() != 2 || output.layout.rank() != 2)
        {
            throw std::invalid_argument("matmul requires rank-2 tensors");
        }
        if (lhs.layout.shape()[1] != rhs.layout.shape()[0])
        {
            throw std::invalid_argument("matmul inner dimensions do not match");
        }
        if (output.layout.shape() != Shape{lhs.layout.shape()[0], rhs.layout.shape()[1]})
        {
            throw std::logic_error("matmul output has the wrong shape");
        }

        const Index rows = lhs.layout.shape()[0];
        const Index inner = lhs.layout.shape()[1];
        const Index columns = rhs.layout.shape()[1];
        for (Index row = 0; row < rows; ++row)
        {
            for (Index column = 0; column < columns; ++column)
            {
                float value = 0.0F;
                for (Index k = 0; k < inner; ++k)
                {
                    const std::array<Index, 2> lhs_coordinates{row, k};
                    const std::array<Index, 2> rhs_coordinates{k, column};
                    value += read_at(
                                 lhs.storage, lhs.layout.offset_from_coordinates(lhs_coordinates)) *
                             read_at(
                                 rhs.storage, rhs.layout.offset_from_coordinates(rhs_coordinates));
                }
                const std::array<Index, 2> output_coordinates{row, column};
                write_at(
                    output.storage,
                    output.layout.offset_from_coordinates(output_coordinates)) = value;
            }
        }
    }

    void reduce_sum(
        const ReadTensorArg input,
        const WriteTensorArg output,
        const std::optional<Index> dim,
        const bool keepdim)
    {
        fill(output, 0.0F);
        for (Index linear = 0; linear < input.layout.numel(); ++linear)
        {
            const auto input_coordinates = input.layout.coordinates_from_linear(linear);
            Coordinates output_coordinates;
            output_coordinates.reserve(as_size(output.layout.rank()));

            if (!dim.has_value())
            {
                output_coordinates.assign(as_size(output.layout.rank()), 0);
            }
            else
            {
                for (Index input_dim = 0; input_dim < input.layout.rank(); ++input_dim)
                {
                    if (input_dim == *dim)
                    {
                        if (keepdim)
                        {
                            output_coordinates.push_back(0);
                        }
                    }
                    else
                    {
                        output_coordinates.push_back(input_coordinates[as_size(input_dim)]);
                    }
                }
            }

            const auto input_offset = input.layout.offset_from_coordinates(input_coordinates);
            const auto output_offset = output.layout.offset_from_coordinates(output_coordinates);
            write_at(output.storage, output_offset) += read_at(input.storage, input_offset);
        }
    }

    void sum_to_shape(const ReadTensorArg input, const WriteTensorArg output)
    {
        if (output.layout.rank() > input.layout.rank())
        {
            throw std::invalid_argument("cannot sum a gradient to a higher-rank shape");
        }
        const Index rank_difference = input.layout.rank() - output.layout.rank();
        for (Index dim = 0; dim < output.layout.rank(); ++dim)
        {
            const auto input_extent = input.layout.shape()[as_size(rank_difference + dim)];
            const auto output_extent = output.layout.shape()[as_size(dim)];
            if (output_extent != 1 && output_extent != input_extent)
            {
                throw std::invalid_argument("gradient cannot be summed to requested shape");
            }
        }

        fill(output, 0.0F);
        for (Index linear = 0; linear < input.layout.numel(); ++linear)
        {
            const auto input_coordinates = input.layout.coordinates_from_linear(linear);
            Coordinates output_coordinates(as_size(output.layout.rank()), 0);
            for (Index dim = 0; dim < output.layout.rank(); ++dim)
            {
                if (output.layout.shape()[as_size(dim)] != 1)
                {
                    output_coordinates[as_size(dim)] =
                        input_coordinates[as_size(rank_difference + dim)];
                }
            }
            const auto input_offset = input.layout.offset_from_coordinates(input_coordinates);
            const auto output_offset = output.layout.offset_from_coordinates(output_coordinates);
            write_at(output.storage, output_offset) += read_at(input.storage, input_offset);
        }
    }

    void slice_scatter(
        const ReadTensorArg gradient,
        const WriteTensorArg output,
        const Index dim,
        const Index start,
        const Index step)
    {
        fill(output, 0.0F);
        for (Index linear = 0; linear < gradient.layout.numel(); ++linear)
        {
            auto coordinates = gradient.layout.coordinates_from_linear(linear);
            const auto gradient_offset = gradient.layout.offset_from_coordinates(coordinates);
            coordinates[as_size(dim)] = start + coordinates[as_size(dim)] * step;
            const auto output_offset = output.layout.offset_from_coordinates(coordinates);
            write_at(output.storage, output_offset) += read_at(gradient.storage, gradient_offset);
        }
    }

    void relu_backward(
        const ReadTensorArg input,
        const ReadTensorArg gradient,
        const WriteTensorArg output)
    {
        require_same_shape(input.layout, gradient.layout, "relu backward");
        require_same_shape(input.layout, output.layout, "relu backward");
        for (Index linear = 0; linear < input.layout.numel(); ++linear)
        {
            const auto coordinates = input.layout.coordinates_from_linear(linear);
            const auto input_offset = input.layout.offset_from_coordinates(coordinates);
            const auto gradient_offset = gradient.layout.offset_from_coordinates(coordinates);
            const auto output_offset = output.layout.offset_from_coordinates(coordinates);
            write_at(output.storage, output_offset) = read_at(input.storage, input_offset) > 0.0F
                                                          ? read_at(gradient.storage, gradient_offset)
                                                          : 0.0F;
        }
    }

} // namespace minitensor::detail
