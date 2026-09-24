#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        template <CpuElement T>
        void dot(const TensorView &lhs, const TensorView &rhs, MutableTensorView output)
        {
            assert(lhs.dtype() == ElementDType<T>::value);
            assert(rhs.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
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
            const auto contraction_extent =
                static_cast<Shape::size_type>(lhs.shape()[0]);
            for (Shape::size_type k = 0; k < contraction_extent; ++k)
            {
                assert(lhs_offset >= 0);
                assert(rhs_offset >= 0);
                accumulator +=
                    lhs_data[static_cast<std::size_t>(lhs_offset)] *
                    rhs_data[static_cast<std::size_t>(rhs_offset)];
                lhs_offset += lhs_stride;
                rhs_offset += rhs_stride;
            }

            output_data[static_cast<std::size_t>(output.layout().offset())] = accumulator;
        }

        template <CpuElement T>
        void gemv(const TensorView &matrix, const TensorView &vector, MutableTensorView output)
        {
            assert(matrix.dtype() == ElementDType<T>::value);
            assert(vector.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
            assert(matrix.shape().rank() == 2);
            assert(vector.shape().rank() == 1);
            assert(output.shape().rank() == 1);
            assert(matrix.shape()[0] == output.shape()[0]);
            assert(matrix.shape()[1] == vector.shape()[0]);

            const T *matrix_data = data<T>(matrix);
            const T *vector_data = data<T>(vector);
            T *output_data = data<T>(output);

            Layout::offset_type matrix_row_offset = matrix.layout().offset();
            const Layout::offset_type vector_base_offset = vector.layout().offset();
            Layout::offset_type output_offset = output.layout().offset();
            const Layout::stride_type matrix_row_stride = matrix.layout().stride(0);
            const Layout::stride_type matrix_contraction_stride = matrix.layout().stride(1);
            const Layout::stride_type vector_stride = vector.layout().stride(0);
            const Layout::stride_type output_stride = output.layout().stride(0);
            const auto row_count = static_cast<Shape::size_type>(matrix.shape()[0]);
            const auto contraction_extent =
                static_cast<Shape::size_type>(matrix.shape()[1]);

            for (Shape::size_type row = 0; row < row_count; ++row)
            {
                Layout::offset_type matrix_offset = matrix_row_offset;
                Layout::offset_type vector_offset = vector_base_offset;
                T accumulator{};
                for (Shape::size_type k = 0; k < contraction_extent; ++k)
                {
                    assert(matrix_offset >= 0);
                    assert(vector_offset >= 0);
                    accumulator +=
                        matrix_data[static_cast<std::size_t>(matrix_offset)] *
                        vector_data[static_cast<std::size_t>(vector_offset)];
                    matrix_offset += matrix_contraction_stride;
                    vector_offset += vector_stride;
                }

                assert(output_offset >= 0);
                output_data[static_cast<std::size_t>(output_offset)] = accumulator;
                matrix_row_offset += matrix_row_stride;
                output_offset += output_stride;
            }
        }

        template <CpuElement T>
        void vector_matrix(
            const TensorView &vector,
            const TensorView &matrix,
            MutableTensorView output)
        {
            assert(vector.dtype() == ElementDType<T>::value);
            assert(matrix.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
            assert(vector.shape().rank() == 1);
            assert(matrix.shape().rank() == 2);
            assert(output.shape().rank() == 1);
            assert(vector.shape()[0] == matrix.shape()[0]);
            assert(matrix.shape()[1] == output.shape()[0]);

            const T *vector_data = data<T>(vector);
            const T *matrix_data = data<T>(matrix);
            T *output_data = data<T>(output);

            const Layout::offset_type vector_base_offset = vector.layout().offset();
            Layout::offset_type matrix_column_offset = matrix.layout().offset();
            Layout::offset_type output_offset = output.layout().offset();
            const Layout::stride_type vector_stride = vector.layout().stride(0);
            const Layout::stride_type matrix_contraction_stride = matrix.layout().stride(0);
            const Layout::stride_type matrix_column_stride = matrix.layout().stride(1);
            const Layout::stride_type output_stride = output.layout().stride(0);
            const auto contraction_extent =
                static_cast<Shape::size_type>(matrix.shape()[0]);
            const auto column_count = static_cast<Shape::size_type>(matrix.shape()[1]);

            for (Shape::size_type column = 0; column < column_count; ++column)
            {
                Layout::offset_type vector_offset = vector_base_offset;
                Layout::offset_type matrix_offset = matrix_column_offset;
                T accumulator{};
                for (Shape::size_type k = 0; k < contraction_extent; ++k)
                {
                    assert(vector_offset >= 0);
                    assert(matrix_offset >= 0);
                    accumulator +=
                        vector_data[static_cast<std::size_t>(vector_offset)] *
                        matrix_data[static_cast<std::size_t>(matrix_offset)];
                    vector_offset += vector_stride;
                    matrix_offset += matrix_contraction_stride;
                }

                assert(output_offset >= 0);
                output_data[static_cast<std::size_t>(output_offset)] = accumulator;
                matrix_column_offset += matrix_column_stride;
                output_offset += output_stride;
            }
        }

        template <CpuElement T>
        void gemm(const TensorView &lhs, const TensorView &rhs, MutableTensorView output)
        {
            assert(lhs.dtype() == ElementDType<T>::value);
            assert(rhs.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
            assert(lhs.shape().rank() == 2);
            assert(rhs.shape().rank() == 2);
            assert(output.shape().rank() == 2);
            assert(lhs.shape()[0] == output.shape()[0]);
            assert(rhs.shape()[1] == output.shape()[1]);
            assert(lhs.shape()[1] == rhs.shape()[0]);

            const T *lhs_data = data<T>(lhs);
            const T *rhs_data = data<T>(rhs);
            T *output_data = data<T>(output);

            Layout::offset_type lhs_row_offset = lhs.layout().offset();
            const Layout::offset_type rhs_base_offset = rhs.layout().offset();
            Layout::offset_type output_row_offset = output.layout().offset();
            const Layout::stride_type lhs_row_stride = lhs.layout().stride(0);
            const Layout::stride_type lhs_contraction_stride = lhs.layout().stride(1);
            const Layout::stride_type rhs_contraction_stride = rhs.layout().stride(0);
            const Layout::stride_type rhs_column_stride = rhs.layout().stride(1);
            const Layout::stride_type output_row_stride = output.layout().stride(0);
            const Layout::stride_type output_column_stride = output.layout().stride(1);
            const auto row_count = static_cast<Shape::size_type>(lhs.shape()[0]);
            const auto contraction_extent = static_cast<Shape::size_type>(lhs.shape()[1]);
            const auto column_count = static_cast<Shape::size_type>(rhs.shape()[1]);

            for (Shape::size_type row = 0; row < row_count; ++row)
            {
                Layout::offset_type rhs_column_offset = rhs_base_offset;
                Layout::offset_type output_offset = output_row_offset;
                for (Shape::size_type column = 0; column < column_count; ++column)
                {
                    Layout::offset_type lhs_offset = lhs_row_offset;
                    Layout::offset_type rhs_offset = rhs_column_offset;
                    T accumulator{};
                    for (Shape::size_type k = 0; k < contraction_extent; ++k)
                    {
                        assert(lhs_offset >= 0);
                        assert(rhs_offset >= 0);
                        accumulator +=
                            lhs_data[static_cast<std::size_t>(lhs_offset)] *
                            rhs_data[static_cast<std::size_t>(rhs_offset)];
                        lhs_offset += lhs_contraction_stride;
                        rhs_offset += rhs_contraction_stride;
                    }

                    assert(output_offset >= 0);
                    output_data[static_cast<std::size_t>(output_offset)] = accumulator;
                    rhs_column_offset += rhs_column_stride;
                    output_offset += output_column_stride;
                }

                lhs_row_offset += lhs_row_stride;
                output_row_offset += output_row_stride;
            }
        }

        template <CpuElement T>
        void matmul_dispatch(
            const TensorView &lhs,
            const TensorView &rhs,
            MutableTensorView output)
        {
            assert(lhs.dtype() == ElementDType<T>::value);
            assert(rhs.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
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
            }
            else if (rhs_rank == 1)
            {
                gemv<T>(lhs, rhs, output);
            }
            else if (lhs_rank == 1)
            {
                vector_matrix<T>(lhs, rhs, output);
            }
            else
            {
                gemm<T>(lhs, rhs, output);
            }
        }

        void run_matmul(
            DeviceRuntime &,
            const Primitive &primitive,
            std::span<const TensorView> inputs,
            MutableTensorView output)
        {
            assert(inputs.size() == 2);
            assert(output.layout().is_contiguous(output.shape()));
            (void)dynamic_cast<const MatmulPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    matmul_dispatch<T>(inputs[0], inputs[1], output);
                });
        }
    }

    void register_linalg_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(
            KernelKey{typeid(MatmulPrimitive), DeviceType::Cpu},
            run_matmul);
    }
}
