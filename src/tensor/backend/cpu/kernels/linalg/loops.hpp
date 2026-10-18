#pragma once

#include <algorithm>
#include <cassert>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/matmul.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
template <CpuElement T>
void dot(const TensorView &lhs, const TensorView &rhs, MutableTensorView output)
{
    assert(lhs.dtype() == dtype_of_v<T>);
    assert(rhs.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(lhs.shape().rank() == 1);
    assert(rhs.shape().rank() == 1);
    assert(output.shape().is_scalar());
    assert(lhs.shape()[0] == rhs.shape()[0]);

    const T *lhs_data = data<T>(lhs);
    const T *rhs_data = data<T>(rhs);
    T *output_data = data<T>(output);

    Layout::offset_type lhs_offset = lhs.layout().offset();
    Layout::offset_type rhs_offset = rhs.layout().offset();
    const Layout::stride_type lhs_stride = lhs.layout().stride(0);
    const Layout::stride_type rhs_stride = rhs.layout().stride(0);

    T accumulator{};
    const auto contraction_extent = static_cast<Shape::size_type>(lhs.shape()[0]);
    for (Shape::size_type k = 0; k < contraction_extent; ++k)
    {
        assert(lhs_offset >= 0);
        assert(rhs_offset >= 0);
        accumulator += lhs_data[static_cast<std::size_t>(lhs_offset)] * rhs_data[static_cast<std::size_t>(rhs_offset)];
        lhs_offset += lhs_stride;
        rhs_offset += rhs_stride;
    }

    assert(output.layout().offset() >= 0);
    output_data[static_cast<std::size_t>(output.layout().offset())] = accumulator;
}

template <CpuElement T>
void gemv(const TensorView &matrix, const TensorView &vector, MutableTensorView output, const MatmulPlan &plan)
{
    assert(matrix.dtype() == dtype_of_v<T>);
    assert(vector.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(matrix.shape().rank() >= 2);
    assert(vector.shape().rank() == 1);
    assert(plan.column_count() == 1);

    const T *matrix_data = data<T>(matrix);
    const T *vector_data = data<T>(vector);
    T *output_data = data<T>(output);

    const MatmulPlan::MatrixStrides &matrix_strides = plan.lhs_strides();
    const MatmulPlan::MatrixStrides &vector_strides = plan.rhs_strides();
    const MatmulPlan::MatrixStrides &output_strides = plan.output_strides();

    plan.for_each_batch([&](MatmulPlan::size_type, const MatmulPlan::BatchOffsets &batch) {
        // A column-oriented matrix makes the output rows contiguous for each
        // contraction element. Accumulating in that order avoids repeatedly
        // traversing a strided matrix column.
        if (matrix_strides.row == 1 && output_strides.row == 1)
        {
            assert(batch.output >= 0);
            T *batch_output = output_data + static_cast<std::size_t>(batch.output);
            std::fill_n(batch_output, plan.row_count(), T{});

            Layout::offset_type matrix_column_offset = batch.lhs;
            Layout::offset_type vector_offset = batch.rhs;
            for (Shape::size_type k = 0; k < plan.contraction_size(); ++k)
            {
                assert(matrix_column_offset >= 0);
                assert(vector_offset >= 0);
                const T vector_value = vector_data[static_cast<std::size_t>(vector_offset)];
                const T *matrix_column = matrix_data + static_cast<std::size_t>(matrix_column_offset);

                for (Shape::size_type row = 0; row < plan.row_count(); ++row)
                {
                    batch_output[row] += matrix_column[row] * vector_value;
                }

                matrix_column_offset += matrix_strides.column;
                vector_offset += vector_strides.row;
            }
            return;
        }

        // General strided path: compute one output element at a time.
        Layout::offset_type matrix_row_offset = batch.lhs;
        Layout::offset_type output_offset = batch.output;

        for (Shape::size_type row = 0; row < plan.row_count(); ++row)
        {
            Layout::offset_type matrix_offset = matrix_row_offset;
            Layout::offset_type vector_offset = batch.rhs;
            T accumulator{};
            for (Shape::size_type k = 0; k < plan.contraction_size(); ++k)
            {
                assert(matrix_offset >= 0);
                assert(vector_offset >= 0);
                accumulator += matrix_data[static_cast<std::size_t>(matrix_offset)] *
                               vector_data[static_cast<std::size_t>(vector_offset)];
                matrix_offset += matrix_strides.column;
                vector_offset += vector_strides.row;
            }

            assert(output_offset >= 0);
            output_data[static_cast<std::size_t>(output_offset)] = accumulator;
            matrix_row_offset += matrix_strides.row;
            output_offset += output_strides.row;
        }
    });
}

template <CpuElement T>
void vector_matrix(const TensorView &vector, const TensorView &matrix, MutableTensorView output, const MatmulPlan &plan)
{
    assert(vector.dtype() == dtype_of_v<T>);
    assert(matrix.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(vector.shape().rank() == 1);
    assert(matrix.shape().rank() >= 2);
    assert(plan.row_count() == 1);

    const T *vector_data = data<T>(vector);
    const T *matrix_data = data<T>(matrix);
    T *output_data = data<T>(output);

    const MatmulPlan::MatrixStrides &vector_strides = plan.lhs_strides();
    const MatmulPlan::MatrixStrides &matrix_strides = plan.rhs_strides();
    const MatmulPlan::MatrixStrides &output_strides = plan.output_strides();

    plan.for_each_batch([&](MatmulPlan::size_type, const MatmulPlan::BatchOffsets &batch) {
        // Row-major matrices expose an entire output row contiguously.
        if (matrix_strides.column == 1 && output_strides.column == 1)
        {
            assert(batch.output >= 0);
            T *batch_output = output_data + static_cast<std::size_t>(batch.output);
            std::fill_n(batch_output, plan.column_count(), T{});

            Layout::offset_type vector_offset = batch.lhs;
            Layout::offset_type matrix_row_offset = batch.rhs;
            for (Shape::size_type k = 0; k < plan.contraction_size(); ++k)
            {
                assert(vector_offset >= 0);
                assert(matrix_row_offset >= 0);
                const T vector_value = vector_data[static_cast<std::size_t>(vector_offset)];
                const T *matrix_row = matrix_data + static_cast<std::size_t>(matrix_row_offset);

                for (Shape::size_type column = 0; column < plan.column_count(); ++column)
                {
                    batch_output[column] += vector_value * matrix_row[column];
                }

                vector_offset += vector_strides.column;
                matrix_row_offset += matrix_strides.row;
            }
            return;
        }

        // General strided path: compute one output element at a time.
        Layout::offset_type matrix_column_offset = batch.rhs;
        Layout::offset_type output_offset = batch.output;

        for (Shape::size_type column = 0; column < plan.column_count(); ++column)
        {
            Layout::offset_type vector_offset = batch.lhs;
            Layout::offset_type matrix_offset = matrix_column_offset;
            T accumulator{};
            for (Shape::size_type k = 0; k < plan.contraction_size(); ++k)
            {
                assert(vector_offset >= 0);
                assert(matrix_offset >= 0);
                accumulator += vector_data[static_cast<std::size_t>(vector_offset)] *
                               matrix_data[static_cast<std::size_t>(matrix_offset)];
                vector_offset += vector_strides.column;
                matrix_offset += matrix_strides.row;
            }

            assert(output_offset >= 0);
            output_data[static_cast<std::size_t>(output_offset)] = accumulator;
            matrix_column_offset += matrix_strides.column;
            output_offset += output_strides.column;
        }
    });
}

template <CpuElement T>
void gemm(const TensorView &lhs, const TensorView &rhs, MutableTensorView output, const MatmulPlan &plan)
{
    assert(lhs.dtype() == dtype_of_v<T>);
    assert(rhs.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(lhs.shape().rank() >= 2);
    assert(rhs.shape().rank() >= 2);

    const T *lhs_data = data<T>(lhs);
    const T *rhs_data = data<T>(rhs);
    T *output_data = data<T>(output);

    const MatmulPlan::MatrixStrides &lhs_strides = plan.lhs_strides();
    const MatmulPlan::MatrixStrides &rhs_strides = plan.rhs_strides();
    const MatmulPlan::MatrixStrides &output_strides = plan.output_strides();

    plan.for_each_batch([&](MatmulPlan::size_type, const MatmulPlan::BatchOffsets &batch) {
        // For a row-major right operand, traverse its rows and the output rows
        // contiguously. This is the common dense GEMM path.
        if (rhs_strides.column == 1 && output_strides.column == 1)
        {
            Layout::offset_type lhs_row_offset = batch.lhs;
            Layout::offset_type output_row_offset = batch.output;

            for (Shape::size_type row = 0; row < plan.row_count(); ++row)
            {
                assert(output_row_offset >= 0);
                T *output_row = output_data + static_cast<std::size_t>(output_row_offset);
                std::fill_n(output_row, plan.column_count(), T{});

                Layout::offset_type lhs_offset = lhs_row_offset;
                Layout::offset_type rhs_row_offset = batch.rhs;
                for (Shape::size_type k = 0; k < plan.contraction_size(); ++k)
                {
                    assert(lhs_offset >= 0);
                    assert(rhs_row_offset >= 0);
                    const T lhs_value = lhs_data[static_cast<std::size_t>(lhs_offset)];
                    const T *rhs_row = rhs_data + static_cast<std::size_t>(rhs_row_offset);

                    for (Shape::size_type column = 0; column < plan.column_count(); ++column)
                    {
                        output_row[column] += lhs_value * rhs_row[column];
                    }

                    lhs_offset += lhs_strides.column;
                    rhs_row_offset += rhs_strides.row;
                }

                lhs_row_offset += lhs_strides.row;
                output_row_offset += output_strides.row;
            }
            return;
        }

        // General strided path: compute one output element at a time.
        Layout::offset_type lhs_row_offset = batch.lhs;
        Layout::offset_type output_row_offset = batch.output;

        for (Shape::size_type row = 0; row < plan.row_count(); ++row)
        {
            Layout::offset_type rhs_column_offset = batch.rhs;
            Layout::offset_type output_offset = output_row_offset;
            for (Shape::size_type column = 0; column < plan.column_count(); ++column)
            {
                Layout::offset_type lhs_offset = lhs_row_offset;
                Layout::offset_type rhs_offset = rhs_column_offset;
                T accumulator{};
                for (Shape::size_type k = 0; k < plan.contraction_size(); ++k)
                {
                    assert(lhs_offset >= 0);
                    assert(rhs_offset >= 0);
                    accumulator +=
                        lhs_data[static_cast<std::size_t>(lhs_offset)] * rhs_data[static_cast<std::size_t>(rhs_offset)];
                    lhs_offset += lhs_strides.column;
                    rhs_offset += rhs_strides.row;
                }

                assert(output_offset >= 0);
                output_data[static_cast<std::size_t>(output_offset)] = accumulator;
                rhs_column_offset += rhs_strides.column;
                output_offset += output_strides.column;
            }

            lhs_row_offset += lhs_strides.row;
            output_row_offset += output_strides.row;
        }
    });
}

template <CpuElement T>
void matmul_dispatch(const TensorView &lhs, const TensorView &rhs, MutableTensorView output)
{
    assert(lhs.dtype() == dtype_of_v<T>);
    assert(rhs.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));

    if (output.shape().numel() == 0)
    {
        return;
    }

    const Shape::size_type lhs_rank = lhs.shape().rank();
    const Shape::size_type rhs_rank = rhs.shape().rank();

    if (lhs_rank == 1 && rhs_rank == 1)
    {
        dot<T>(lhs, rhs, output);
        return;
    }

    const MatmulPlan plan(lhs.shape(), lhs.layout(), rhs.shape(), rhs.layout(), output.shape(), output.layout());

    if (rhs_rank == 1)
    {
        gemv<T>(lhs, rhs, output, plan);
    }
    else if (lhs_rank == 1)
    {
        vector_matrix<T>(lhs, rhs, output, plan);
    }
    else
    {
        gemm<T>(lhs, rhs, output, plan);
    }
}
} // namespace minitensor::detail::cpu
