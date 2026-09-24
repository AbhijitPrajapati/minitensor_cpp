#include <minitensor/types.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tensor/backend/cpu/iteration/matmul.hpp"
#include "tensor/storage/layout.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_matmul_plan_test()
    {
        using detail::Layout;
        using detail::cpu::MatmulPlan;

        const MatmulPlan matrix_plan{
            Shape{2, 3}, Layout{{4, 1}, 5},
            Shape{3, 4}, Layout{{1, 3}, 7},
            Shape{2, 4}, Layout{{4, 1}, 11}};
        expect(matrix_plan.batch_count() == 1,
               "an unbatched matmul plan contains one batch");
        expect(matrix_plan.row_count() == 2 &&
                   matrix_plan.contraction_size() == 3 &&
                   matrix_plan.column_count() == 4,
               "a matmul plan exposes its logical M, K, and N dimensions");
        expect(matrix_plan.lhs_strides() == MatmulPlan::MatrixStrides{4, 1} &&
                   matrix_plan.rhs_strides() == MatmulPlan::MatrixStrides{1, 3} &&
                   matrix_plan.output_strides() == MatmulPlan::MatrixStrides{4, 1},
               "a matmul plan preserves strided matrix dimensions");

        std::vector<MatmulPlan::BatchOffsets> matrix_offsets;
        matrix_plan.for_each_batch(
            [&matrix_offsets](Shape::size_type batch, const MatmulPlan::BatchOffsets &offsets)
            {
                expect(batch == 0, "an unbatched matmul plan reports batch index zero");
                matrix_offsets.push_back(offsets);
            });
        expect(matrix_offsets == std::vector<MatmulPlan::BatchOffsets>{{5, 7, 11}},
               "an unbatched matmul plan preserves layout offsets");

        const MatmulPlan vector_plan{
            Shape{3}, Layout{{2}, 3},
            Shape{3}, Layout{{4}, 5},
            Shape{}, Layout{std::vector<Layout::stride_type>{}, 7}};
        expect(vector_plan.row_count() == 1 &&
                   vector_plan.contraction_size() == 3 &&
                   vector_plan.column_count() == 1,
               "a vector-vector plan normalizes both synthetic matrix dimensions");
        expect(vector_plan.lhs_strides() == MatmulPlan::MatrixStrides{0, 2} &&
                   vector_plan.rhs_strides() == MatmulPlan::MatrixStrides{4, 0} &&
                   vector_plan.output_strides() == MatmulPlan::MatrixStrides{0, 0},
               "a vector-vector plan gives synthetic dimensions zero strides");

        const MatmulPlan broadcast_plan{
            Shape{2, 1, 3, 4}, Layout{{12, 12, 4, 1}, 5},
            Shape{5, 4, 6}, Layout{{24, 6, 1}, 7},
            Shape{2, 5, 3, 6}, Layout{{90, 18, 6, 1}, 11}};
        expect(broadcast_plan.batch_count() == 10,
               "a matmul plan counts every broadcasted batch");

        std::vector<MatmulPlan::BatchOffsets> broadcast_offsets;
        broadcast_plan.for_each_batch(
            [&broadcast_offsets](Shape::size_type batch, const MatmulPlan::BatchOffsets &offsets)
            {
                expect(batch == broadcast_offsets.size(),
                       "a matmul plan reports batches in row-major order");
                broadcast_offsets.push_back(offsets);
            });
        const std::vector<MatmulPlan::BatchOffsets> expected_broadcast_offsets{
            {5, 7, 11},
            {5, 31, 29},
            {5, 55, 47},
            {5, 79, 65},
            {5, 103, 83},
            {17, 7, 101},
            {17, 31, 119},
            {17, 55, 137},
            {17, 79, 155},
            {17, 103, 173}};
        expect(broadcast_offsets == expected_broadcast_offsets,
               "a matmul plan aligns missing and singleton batch dimensions");

        const MatmulPlan vector_matrix_plan{
            Shape{4}, Layout{{2}, 3},
            Shape{2, 4, 5}, Layout{{20, 5, 1}, 7},
            Shape{2, 5}, Layout{{5, 1}, 11}};
        std::vector<MatmulPlan::BatchOffsets> vector_matrix_offsets;
        vector_matrix_plan.for_each_batch(
            [&vector_matrix_offsets](Shape::size_type, const MatmulPlan::BatchOffsets &offsets)
            {
                vector_matrix_offsets.push_back(offsets);
            });
        expect(vector_matrix_offsets ==
                   std::vector<MatmulPlan::BatchOffsets>{{3, 7, 11}, {3, 27, 16}},
               "a vector operand is reused across every matrix batch");
        expect(vector_matrix_plan.output_strides() == MatmulPlan::MatrixStrides{0, 1},
               "a vector-matrix output has a synthetic zero row stride");

        const MatmulPlan matrix_vector_plan{
            Shape{2, 3, 4}, Layout{{12, 4, 1}, 13},
            Shape{4}, Layout{{2}, 17},
            Shape{2, 3}, Layout{{3, 1}, 19}};
        std::vector<MatmulPlan::BatchOffsets> matrix_vector_offsets;
        matrix_vector_plan.for_each_batch(
            [&matrix_vector_offsets](Shape::size_type, const MatmulPlan::BatchOffsets &offsets)
            {
                matrix_vector_offsets.push_back(offsets);
            });
        expect(matrix_vector_offsets ==
                   std::vector<MatmulPlan::BatchOffsets>{{13, 17, 19}, {25, 17, 22}},
               "a right-hand vector is reused across every matrix batch");
        expect(matrix_vector_plan.rhs_strides() == MatmulPlan::MatrixStrides{2, 0} &&
                   matrix_vector_plan.output_strides() == MatmulPlan::MatrixStrides{1, 0},
               "a matrix-vector plan gives synthetic column dimensions zero strides");

        const MatmulPlan empty_batch_plan{
            Shape{0, 3, 4}, Layout::contiguous(Shape{0, 3, 4}),
            Shape{4, 5}, Layout::contiguous(Shape{4, 5}),
            Shape{0, 3, 5}, Layout::contiguous(Shape{0, 3, 5})};
        std::size_t empty_batch_visits = 0;
        empty_batch_plan.for_each_batch(
            [&empty_batch_visits](Shape::size_type, const MatmulPlan::BatchOffsets &)
            {
                ++empty_batch_visits;
            });
        expect(empty_batch_plan.batch_count() == 0 && empty_batch_visits == 0,
               "an empty batch produces no matmul slices");

        expect_throws<std::invalid_argument>(
            []
            {
                (void)MatmulPlan{
                    Shape{2, 3}, Layout{{1}},
                    Shape{3, 4}, Layout::contiguous(Shape{3, 4}),
                    Shape{2, 4}, Layout::contiguous(Shape{2, 4})};
            },
            "a matmul plan rejects a shape and layout rank mismatch");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)MatmulPlan{
                    Shape{2, 3}, Layout::contiguous(Shape{2, 3}),
                    Shape{3, 4}, Layout::contiguous(Shape{3, 4}),
                    Shape{2, 5}, Layout::contiguous(Shape{2, 5})};
            },
            "a matmul plan rejects an incorrect output shape");
    }
}
