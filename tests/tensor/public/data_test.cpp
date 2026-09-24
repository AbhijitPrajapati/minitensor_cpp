#include <minitensor/data.hpp>
#include <minitensor/ops.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_data_test()
    {
        std::array<float, 6> source{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        const std::array<float, 6> expected_source = source;
        const Tensor matrix = from_data(source, Shape{2, 3});

        expect(matrix.shape() == Shape{2, 3}, "from_data preserves the requested shape");
        expect(matrix.dtype() == DType::Float32, "from_data creates a Float32 tensor");
        expect(matrix.device() == Device::cpu(), "from_data uses the default device");

        source[0] = -100.0F;
        const auto matrix_values = to_vector(matrix);
        expect(std::ranges::equal(matrix_values, expected_source),
               "from_data copies its input and to_vector preserves logical element order");

        expect(std::ranges::equal(to_vector(reshape(matrix, Shape{3, 2})), expected_source),
               "a contiguous reshape preserves logical element order");

        constexpr std::array<Axis, 2> swapped_axes{1, 0};
        const std::array<float, 6> expected_permuted{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
        const Tensor permuted_matrix = permute(matrix, swapped_axes);
        expect(std::ranges::equal(to_vector(permuted_matrix), expected_permuted),
               "to_vector reads a permuted view in its logical element order");
        expect(std::ranges::equal(to_vector(contiguous(permuted_matrix)), expected_permuted),
               "contiguous materializes a strided input in logical element order");
        expect(std::ranges::equal(
                   to_vector(reshape(permuted_matrix, Shape{2, 3})), expected_permuted),
               "reshape preserves logical element order for a noncontiguous input");
        expect(std::ranges::equal(to_vector(flatten(permuted_matrix)), expected_permuted),
               "flatten preserves the logical order of a noncontiguous input");
        expect(std::ranges::equal(to_vector(transpose(matrix)), expected_permuted),
               "transpose reverses matrix axes through permutation");
        expect(std::ranges::equal(
                   to_vector(squeeze(unsqueeze(matrix, -1))), expected_source),
               "squeeze and unsqueeze preserve element order through reshape");

        const std::array<float, 3> row_values{7.0F, 8.0F, 9.0F};
        const std::array<float, 6> expected_broadcast{7.0F, 8.0F, 9.0F, 7.0F, 8.0F, 9.0F};
        const Tensor broadcasted_row = broadcast_to(from_data(row_values, Shape{1, 3}), Shape{2, 3});
        expect(std::ranges::equal(to_vector(broadcasted_row), expected_broadcast),
               "to_vector reads broadcasted storage in logical element order");

        const std::array<float, 1> scalar_source{-3.25F};
        const Tensor scalar = from_data(scalar_source, Shape{});
        expect(item(scalar) == -3.25F, "item returns the value of a scalar tensor");

        const Tensor ranked_singleton = full(Shape{1, 1}, 7.5F);
        expect(item(ranked_singleton) == 7.5F,
               "item evaluates and accepts any tensor containing exactly one element");

        const Tensor empty = from_data(std::span<const float>{}, Shape{2, 0, 3});
        expect(to_vector(empty).empty(), "data transfer supports empty tensors");

        const std::array<float, 6> expected_zeros{};
        const std::array<float, 6> expected_ones{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        const std::array<float, 6> expected_full_like{-2.0F, -2.0F, -2.0F, -2.0F, -2.0F, -2.0F};
        expect(std::ranges::equal(to_vector(zeros_like(matrix)), expected_zeros),
               "zeros_like composes full with the input metadata and zero value");
        expect(std::ranges::equal(to_vector(ones(Shape{2, 3})), expected_ones),
               "ones composes full with the requested shape and unit value");
        expect(std::ranges::equal(to_vector(full_like(matrix, -2.0F)), expected_full_like),
               "full_like composes full with the input shape and requested value");

        const std::array<float, 2> lhs_values{1.0F, 2.0F};
        const std::array<float, 3> rhs_values{10.0F, 20.0F, 30.0F};
        const Tensor added = from_data(lhs_values, Shape{2, 1}) +
                             from_data(rhs_values, Shape{1, 3});
        const std::array<float, 6> expected_sum{11.0F, 21.0F, 31.0F, 12.0F, 22.0F, 32.0F};
        expect(std::ranges::equal(to_vector(added), expected_sum),
               "to_vector evaluates a graph whose materialized leaves came from host data");

        const Tensor negated = -permuted_matrix;
        const std::array<float, 6> expected_negated{-1.0F, -4.0F, -2.0F, -5.0F, -3.0F, -6.0F};
        expect(std::ranges::equal(to_vector(negated), expected_negated),
               "elementwise negation computes additive inverses from a strided input");

        const std::array<float, 6> expected_doubled_permutation{2.0F, 8.0F, 4.0F, 10.0F, 6.0F, 12.0F};
        expect(std::ranges::equal(
                   to_vector(permuted_matrix + permuted_matrix),
                   expected_doubled_permutation),
               "binary elementwise kernels support two strided inputs");

        const std::array<float, 6> expected_scalar_sum{3.0F, 6.0F, 4.0F, 7.0F, 5.0F, 8.0F};
        expect(std::ranges::equal(
                   to_vector(permuted_matrix + full(Shape{}, 2.0F)),
                   expected_scalar_sum),
               "binary elementwise kernels support scalar broadcasting with a strided input");

        const std::array<float, 3> positive_values{1.0F, 2.0F, 4.0F};
        const Tensor positive_input = from_data(positive_values, Shape{3});
        const std::array<float, 3> expected_exp{
            std::exp(1.0F), std::exp(2.0F), std::exp(4.0F)};
        const std::array<float, 3> expected_log{
            0.0F, std::log(2.0F), std::log(4.0F)};
        const std::array<float, 3> expected_sqrt{1.0F, std::sqrt(2.0F), 2.0F};
        const std::array<float, 3> expected_tanh{
            std::tanh(1.0F), std::tanh(2.0F), std::tanh(4.0F)};
        expect_near(to_vector(exp(positive_input)), expected_exp, 1.0E-6F,
                    "exp computes elementwise exponentials");
        expect_near(to_vector(log(positive_input)), expected_log, 1.0E-6F,
                    "log computes elementwise natural logarithms");
        expect_near(to_vector(sqrt(positive_input)), expected_sqrt, 1.0E-6F,
                    "sqrt computes elementwise square roots");
        expect_near(to_vector(tanh(positive_input)), expected_tanh, 1.0E-6F,
                    "tanh computes elementwise hyperbolic tangents");

        const Tensor subtracted = from_data(lhs_values, Shape{2, 1}) -
                                  from_data(rhs_values, Shape{1, 3});
        const std::array<float, 6> expected_difference{-9.0F, -19.0F, -29.0F, -8.0F, -18.0F, -28.0F};
        expect(std::ranges::equal(to_vector(subtracted), expected_difference),
               "elementwise subtraction computes broadcasted differences");

        const Tensor multiplied = from_data(lhs_values, Shape{2, 1}) *
                                  from_data(rhs_values, Shape{1, 3});
        const std::array<float, 6> expected_product{10.0F, 20.0F, 30.0F, 20.0F, 40.0F, 60.0F};
        expect(std::ranges::equal(to_vector(multiplied), expected_product),
               "elementwise multiplication computes broadcasted products");

        const Tensor divided = from_data(lhs_values, Shape{2, 1}) /
                               from_data(rhs_values, Shape{1, 3});
        const std::array<float, 6> expected_quotient{
            1.0F / 10.0F, 1.0F / 20.0F, 1.0F / 30.0F,
            2.0F / 10.0F, 2.0F / 20.0F, 2.0F / 30.0F};
        expect(std::ranges::equal(to_vector(divided), expected_quotient),
               "elementwise division computes broadcasted quotients");

        const std::array<float, 3> dot_lhs_values{1.0F, 2.0F, 3.0F};
        const std::array<float, 3> dot_rhs_values{4.0F, 5.0F, 6.0F};
        expect(item(matmul(
                   from_data(dot_lhs_values, Shape{3}),
                   from_data(dot_rhs_values, Shape{3}))) == 32.0F,
               "vector-vector matmul computes a dot product");

        const std::array<float, 3> vector_values{1.0F, 0.0F, -1.0F};
        const std::array<float, 2> expected_matrix_vector{-2.0F, -2.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(matrix, from_data(vector_values, Shape{3}))),
                   expected_matrix_vector),
               "matrix-vector matmul computes a GEMV product");

        const std::array<float, 2> left_vector_values{1.0F, 2.0F};
        const std::array<float, 3> expected_vector_matrix{9.0F, 12.0F, 15.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(
                       from_data(left_vector_values, Shape{2}),
                       matrix)),
                   expected_vector_matrix),
               "vector-matrix matmul contracts the vector with matrix rows");

        const std::array<float, 6> lhs_storage{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
        const std::array<float, 6> rhs_storage{7.0F, 9.0F, 11.0F, 8.0F, 10.0F, 12.0F};
        const Tensor strided_lhs = transpose(from_data(lhs_storage, Shape{3, 2}));
        const Tensor strided_rhs = transpose(from_data(rhs_storage, Shape{2, 3}));
        const std::array<float, 4> expected_matrix_matrix{58.0F, 64.0F, 139.0F, 154.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(strided_lhs, strided_rhs)),
                   expected_matrix_matrix),
               "matrix-matrix matmul honors strides in both inputs");

        const std::array<float, 12> batched_matrix_values{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
            -1.0F, 0.0F, 1.0F,
            2.0F, -2.0F, 3.0F};
        const Tensor batched_matrix = from_data(batched_matrix_values, Shape{2, 2, 3});
        const std::array<float, 3> batched_gemv_vector_values{2.0F, -1.0F, 3.0F};
        const std::array<float, 4> expected_batched_gemv{9.0F, 21.0F, 1.0F, 15.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(
                       batched_matrix,
                       from_data(batched_gemv_vector_values, Shape{3}))),
                   expected_batched_gemv),
               "batched matrix-vector matmul reuses the vector across batches");

        const std::array<float, 2> batched_vector_matrix_values{2.0F, -1.0F};
        const std::array<float, 6> expected_batched_vector_matrix{
            -2.0F, -1.0F, 0.0F,
            -4.0F, 2.0F, -1.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(
                       from_data(batched_vector_matrix_values, Shape{2}),
                       batched_matrix)),
                   expected_batched_vector_matrix),
               "batched vector-matrix matmul reuses the vector across batches");

        const std::array<float, 8> batched_gemm_lhs_values{
            1.0F, 2.0F,
            3.0F, 4.0F,
            -1.0F, 0.0F,
            2.0F, 3.0F};
        const std::array<float, 8> batched_gemm_rhs_values{
            5.0F, 6.0F,
            7.0F, 8.0F,
            4.0F, -2.0F,
            1.0F, 5.0F};
        const std::array<float, 8> expected_batched_gemm{
            19.0F, 22.0F,
            43.0F, 50.0F,
            -4.0F, 2.0F,
            11.0F, 11.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(
                       from_data(batched_gemm_lhs_values, Shape{2, 2, 2}),
                       from_data(batched_gemm_rhs_values, Shape{2, 2, 2}))),
                   expected_batched_gemm),
               "batched matrix-matrix matmul computes each corresponding batch");

        const std::array<float, 4> broadcast_matmul_lhs_values{1.0F, 2.0F, 3.0F, 4.0F};
        const std::array<float, 6> broadcast_matmul_rhs_values{
            10.0F, 1.0F,
            2.0F, 3.0F,
            -1.0F, 5.0F};
        const std::array<float, 6> expected_broadcast_matmul{
            12.0F, 8.0F, 9.0F,
            34.0F, 18.0F, 17.0F};
        expect(std::ranges::equal(
                   to_vector(matmul(
                       from_data(broadcast_matmul_lhs_values, Shape{2, 1, 1, 2}),
                       from_data(broadcast_matmul_rhs_values, Shape{3, 2, 1}))),
                   expected_broadcast_matmul),
               "batched matrix-matrix matmul broadcasts each operand independently");

        constexpr std::array<Axis, 3> batched_matrix_permutation{1, 2, 0};
        const std::array<float, 8> strided_batched_lhs_storage{
            1.0F, 3.0F, -1.0F, 2.0F,
            2.0F, 4.0F, 0.0F, 3.0F};
        const std::array<float, 8> strided_batched_rhs_storage{
            5.0F, 7.0F, 4.0F, 1.0F,
            6.0F, 8.0F, -2.0F, 5.0F};
        const Tensor strided_batched_lhs = permute(
            from_data(strided_batched_lhs_storage, Shape{2, 2, 2}),
            batched_matrix_permutation);
        const Tensor strided_batched_rhs = permute(
            from_data(strided_batched_rhs_storage, Shape{2, 2, 2}),
            batched_matrix_permutation);
        expect(std::ranges::equal(
                   to_vector(matmul(strided_batched_lhs, strided_batched_rhs)),
                   expected_batched_gemm),
               "batched matrix-matrix matmul honors permuted batch and matrix strides");

        const std::array<float, 2> expected_zero_matrix_vector{};
        const std::array<float, 3> expected_zero_vector_matrix{};
        const std::array<float, 6> expected_zero_matrix{};
        expect(item(matmul(full(Shape{0}, 1.0F), full(Shape{0}, 1.0F))) == 0.0F,
               "an empty dot product produces the additive identity");
        expect(std::ranges::equal(
                   to_vector(matmul(full(Shape{2, 0}, 1.0F), full(Shape{0}, 1.0F))),
                   expected_zero_matrix_vector),
               "matrix-vector matmul handles an empty contraction dimension");
        expect(std::ranges::equal(
                   to_vector(matmul(full(Shape{0}, 1.0F), full(Shape{0, 3}, 1.0F))),
                   expected_zero_vector_matrix),
               "vector-matrix matmul handles an empty contraction dimension");
        expect(std::ranges::equal(
                   to_vector(matmul(full(Shape{2, 0}, 1.0F), full(Shape{0, 3}, 1.0F))),
                   expected_zero_matrix),
               "matrix-matrix matmul handles an empty contraction dimension");

        const std::array<float, 4> expected_batched_zero_matrix_vector{};
        const std::array<float, 6> expected_batched_zero_vector_matrix{};
        const std::array<float, 12> expected_batched_zero_matrix{};
        expect(std::ranges::equal(
                   to_vector(matmul(full(Shape{2, 2, 0}, 1.0F), full(Shape{0}, 1.0F))),
                   expected_batched_zero_matrix_vector),
               "batched matrix-vector matmul writes zero for an empty contraction");
        expect(std::ranges::equal(
                   to_vector(matmul(full(Shape{0}, 1.0F), full(Shape{2, 0, 3}, 1.0F))),
                   expected_batched_zero_vector_matrix),
               "batched vector-matrix matmul writes zero for an empty contraction");
        expect(std::ranges::equal(
                   to_vector(matmul(full(Shape{2, 2, 0}, 1.0F), full(Shape{1, 0, 3}, 1.0F))),
                   expected_batched_zero_matrix),
               "batched matrix-matrix matmul writes zero for an empty contraction");
        expect(to_vector(matmul(
                   full(Shape{0, 2, 3}, 1.0F),
                   full(Shape{3, 4}, 1.0F))).empty(),
               "matmul accepts an empty batch without accessing storage");

        constexpr std::array<Axis, 1> last_axis{-1};
        const std::array<float, 2> expected_row_sums{6.0F, 15.0F};
        expect(std::ranges::equal(to_vector(sum(matrix, last_axis)), expected_row_sums),
               "sum reduces a selected axis and accepts its negative spelling");
        expect(std::ranges::equal(to_vector(sum(matrix, last_axis, true)), expected_row_sums),
               "sum with keep_dim preserves reduction results while retaining the axis");
        expect(item(sum(matrix)) == 21.0F,
               "sum without explicit axes reduces all input elements");

        constexpr std::array<Axis, 1> leading_axis{0};
        const std::array<float, 2> expected_permuted_column_sums{6.0F, 15.0F};
        expect(std::ranges::equal(
                   to_vector(sum(permuted_matrix, leading_axis)),
                   expected_permuted_column_sums),
               "sum handles a leading reduction axis on a strided input");

        constexpr std::array<Axis, 1> permuted_last_axis{1};
        const std::array<float, 3> expected_permuted_row_sums{5.0F, 7.0F, 9.0F};
        expect(std::ranges::equal(
                   to_vector(sum(permuted_matrix, permuted_last_axis)),
                   expected_permuted_row_sums),
               "sum handles a noncontiguous reduced dimension");

        const std::array<float, 3> expected_broadcast_sums{14.0F, 16.0F, 18.0F};
        expect(std::ranges::equal(
                   to_vector(sum(broadcasted_row, leading_axis)),
                   expected_broadcast_sums),
               "sum honors zero strides in a broadcasted input");

        const std::array<float, 12> volume_values{
            1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
            7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
        constexpr std::array<Axis, 2> nonadjacent_axes{0, 2};
        const std::array<float, 3> expected_nonadjacent_sums{18.0F, 26.0F, 34.0F};
        expect(std::ranges::equal(
                   to_vector(sum(
                       from_data(volume_values, Shape{2, 3, 2}),
                       nonadjacent_axes)),
                   expected_nonadjacent_sums),
               "sum handles multiple nonadjacent reduction axes");

        constexpr std::array<Axis, 1> zero_length_axis{1};
        const std::array<float, 6> expected_empty_sums{};
        expect(std::ranges::equal(to_vector(sum(empty, zero_length_axis)), expected_empty_sums),
               "sum produces the additive identity when a reduction axis is empty");

        expect_throws<std::invalid_argument>(
            []
            {
                const std::array<float, 2> wrong_size{1.0F, 2.0F};
                (void)from_data(wrong_size, Shape{3});
            },
            "from_data rejects data whose element count does not match the shape");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                (void)item(matrix);
            },
            "item rejects tensors that do not contain exactly one element");
    }
}
