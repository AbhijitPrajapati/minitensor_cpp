#include "kernels/kernels.hpp"
#include "kernels/iteration.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace minitensor::detail::kernel
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

    void fill(const WriteTensorArg output, const float value)
    {
        const ElementwiseIterator iterator(output.layout, {});
        iterator.for_each(
            [&](const Index output_offset, const std::span<const Index>)
            { write_at(output.storage, output_offset) = value; });
    }

    void copy(const ReadTensorArg input, const WriteTensorArg output)
    {
        require_same_shape(input.layout, output.layout, "copy");
        const ElementwiseIterator iterator(output.layout, {input.layout});
        iterator.for_each(
            [&](const Index output_offset, const std::span<const Index> input_offsets)
            {
                write_at(output.storage, output_offset) =
                    read_at(input.storage, input_offsets[0]);
            });
    }

    void binary(
        const ReadTensorArg lhs,
        const ReadTensorArg rhs,
        const WriteTensorArg output,
        const BinaryKernel operation)
    {
        const ElementwiseIterator iterator(output.layout, {lhs.layout, rhs.layout});
        iterator.for_each(
            [&](const Index output_offset, const std::span<const Index> input_offsets)
            {
                write_at(output.storage, output_offset) = apply_binary(
                    read_at(lhs.storage, input_offsets[0]),
                    read_at(rhs.storage, input_offsets[1]),
                    operation);
            });
    }

    void unary(
        const ReadTensorArg input,
        const WriteTensorArg output,
        const UnaryKernel operation)
    {
        require_same_shape(input.layout, output.layout, "unary kernel");
        const ElementwiseIterator iterator(output.layout, {input.layout});
        iterator.for_each(
            [&](const Index output_offset, const std::span<const Index> input_offsets)
            {
                write_at(output.storage, output_offset) = apply_unary(
                    read_at(input.storage, input_offsets[0]), operation);
            });
    }

    void matrix_multiply(
        const ReadTensorArg lhs,
        const ReadTensorArg rhs,
        const WriteTensorArg output)
    {
        const MatrixMultiplyPlan plan(lhs.layout, rhs.layout, output.layout);
        fill(output, 0.0F);
        plan.for_each(
            [&](const Index output_offset, const Index lhs_offset, const Index rhs_offset)
            {
                write_at(output.storage, output_offset) +=
                    read_at(lhs.storage, lhs_offset) * read_at(rhs.storage, rhs_offset);
            });
    }

    void reduce_sum(
        const ReadTensorArg input,
        const WriteTensorArg output,
        const std::optional<Index> dim,
        const bool keepdim)
    {
        const ReductionIterator iterator(input.layout, output.layout, dim, keepdim);
        fill(output, 0.0F);
        iterator.for_each(
            [&](const Index output_offset, const Index input_offset)
            {
                write_at(output.storage, output_offset) += read_at(input.storage, input_offset);
            });
    }

    void sum_to_shape(const ReadTensorArg input, const WriteTensorArg output)
    {
        const auto iterator = ReductionIterator::to_shape(input.layout, output.layout);
        fill(output, 0.0F);
        iterator.for_each(
            [&](const Index output_offset, const Index input_offset)
            {
                write_at(output.storage, output_offset) += read_at(input.storage, input_offset);
            });
    }

    void relu_backward(
        const ReadTensorArg input,
        const ReadTensorArg gradient,
        const WriteTensorArg output)
    {
        require_same_shape(input.layout, gradient.layout, "relu backward");
        require_same_shape(input.layout, output.layout, "relu backward");
        const ElementwiseIterator iterator(output.layout, {input.layout, gradient.layout});
        iterator.for_each(
            [&](const Index output_offset, const std::span<const Index> input_offsets)
            {
                write_at(output.storage, output_offset) =
                    read_at(input.storage, input_offsets[0]) > 0.0F
                        ? read_at(gradient.storage, input_offsets[1])
                        : 0.0F;
            });
    }

} // namespace minitensor::detail::kernel
