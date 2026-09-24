#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/ops.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_autograd_test()
    {
        const std::array<float, 2> lhs_values{1.0F, 2.0F};
        const std::array<float, 3> rhs_values{10.0F, 20.0F, 30.0F};
        const Tensor lhs = from_data(lhs_values, Shape{2, 1});
        const Tensor rhs = from_data(rhs_values, Shape{1, 3});
        const Tensor loss = sum(lhs * rhs + lhs);
        const std::array<Tensor, 2> loss_inputs{lhs, rhs};
        const std::vector<Tensor> loss_gradients = grad(loss, loss_inputs);

        expect(loss_gradients.size() == 2,
               "grad returns one gradient for each requested input");
        const std::array<float, 2> expected_lhs_gradient{63.0F, 63.0F};
        const std::array<float, 3> expected_rhs_gradient{3.0F, 3.0F, 3.0F};
        expect(loss_gradients[0].shape() == lhs.shape() &&
                   std::ranges::equal(to_vector(loss_gradients[0]), expected_lhs_gradient),
               "grad accumulates broadcasted add and multiply contributions");
        expect(loss_gradients[1].shape() == rhs.shape() &&
                   std::ranges::equal(to_vector(loss_gradients[1]), expected_rhs_gradient),
               "grad reduces a broadcasted multiply contribution to its input shape");

        const std::array<float, 6> matmul_lhs_values{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F};
        const std::array<float, 6> matmul_rhs_values{
            7.0F, 8.0F,
            9.0F, 10.0F,
            11.0F, 12.0F};
        const Tensor matmul_lhs = from_data(matmul_lhs_values, Shape{2, 3});
        const Tensor matmul_rhs = from_data(matmul_rhs_values, Shape{3, 2});
        const std::array<Tensor, 2> matmul_inputs{matmul_lhs, matmul_rhs};
        const std::vector<Tensor> matmul_gradients =
            grad(sum(matmul(matmul_lhs, matmul_rhs)), matmul_inputs);
        const std::array<float, 6> expected_matmul_lhs_gradient{
            15.0F, 19.0F, 23.0F,
            15.0F, 19.0F, 23.0F};
        const std::array<float, 6> expected_matmul_rhs_gradient{
            5.0F, 5.0F,
            7.0F, 7.0F,
            9.0F, 9.0F};
        expect(std::ranges::equal(
                   to_vector(matmul_gradients[0]),
                   expected_matmul_lhs_gradient),
               "matrix-matrix matmul propagates the lhs cotangent");
        expect(std::ranges::equal(
                   to_vector(matmul_gradients[1]),
                   expected_matmul_rhs_gradient),
               "matrix-matrix matmul propagates the rhs cotangent");

        const std::array<float, 3> dot_autograd_lhs_values{1.0F, 2.0F, 3.0F};
        const std::array<float, 3> dot_autograd_rhs_values{4.0F, 5.0F, 6.0F};
        const Tensor dot_autograd_lhs = from_data(dot_autograd_lhs_values, Shape{3});
        const Tensor dot_autograd_rhs = from_data(dot_autograd_rhs_values, Shape{3});
        const std::array<Tensor, 2> dot_autograd_inputs{
            dot_autograd_lhs,
            dot_autograd_rhs};
        const std::vector<Tensor> dot_autograd_gradients = grad(
            matmul(dot_autograd_lhs, dot_autograd_rhs),
            dot_autograd_inputs);
        expect(std::ranges::equal(
                   to_vector(dot_autograd_gradients[0]),
                   dot_autograd_rhs_values) &&
                   std::ranges::equal(
                       to_vector(dot_autograd_gradients[1]),
                       dot_autograd_lhs_values),
               "vector-vector matmul removes both synthetic VJP dimensions");

        const std::array<float, 12> batched_matmul_matrix_values{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
            -1.0F, 0.0F, 1.0F,
            2.0F, -2.0F, 3.0F};
        const std::array<float, 3> batched_matmul_vector_values{2.0F, -1.0F, 3.0F};
        const Tensor batched_matmul_matrix = from_data(
            batched_matmul_matrix_values,
            Shape{2, 2, 3});
        const Tensor batched_matmul_vector = from_data(
            batched_matmul_vector_values,
            Shape{3});
        const std::array<Tensor, 2> batched_matrix_vector_inputs{
            batched_matmul_matrix,
            batched_matmul_vector};
        const std::vector<Tensor> batched_matrix_vector_gradients = grad(
            sum(matmul(batched_matmul_matrix, batched_matmul_vector)),
            batched_matrix_vector_inputs);
        const std::array<float, 12> expected_batched_matrix_gradient{
            2.0F, -1.0F, 3.0F,
            2.0F, -1.0F, 3.0F,
            2.0F, -1.0F, 3.0F,
            2.0F, -1.0F, 3.0F};
        const std::array<float, 3> expected_batched_vector_gradient{6.0F, 5.0F, 13.0F};
        expect(std::ranges::equal(
                   to_vector(batched_matrix_vector_gradients[0]),
                   expected_batched_matrix_gradient),
               "batched matrix-vector matmul restores the lhs VJP shape");
        expect(std::ranges::equal(
                   to_vector(batched_matrix_vector_gradients[1]),
                   expected_batched_vector_gradient),
               "batched matrix-vector matmul reduces the shared vector VJP across batches");

        const std::array<float, 2> batched_left_vector_values{2.0F, -1.0F};
        const Tensor batched_left_vector = from_data(batched_left_vector_values, Shape{2});
        const std::array<Tensor, 2> batched_vector_matrix_inputs{
            batched_left_vector,
            batched_matmul_matrix};
        const std::vector<Tensor> batched_vector_matrix_gradients = grad(
            sum(matmul(batched_left_vector, batched_matmul_matrix)),
            batched_vector_matrix_inputs);
        const std::array<float, 2> expected_batched_left_vector_gradient{6.0F, 18.0F};
        const std::array<float, 12> expected_batched_rhs_matrix_gradient{
            2.0F, 2.0F, 2.0F,
            -1.0F, -1.0F, -1.0F,
            2.0F, 2.0F, 2.0F,
            -1.0F, -1.0F, -1.0F};
        expect(std::ranges::equal(
                   to_vector(batched_vector_matrix_gradients[0]),
                   expected_batched_left_vector_gradient),
               "batched vector-matrix matmul reduces the shared vector VJP across batches");
        expect(std::ranges::equal(
                   to_vector(batched_vector_matrix_gradients[1]),
                   expected_batched_rhs_matrix_gradient),
               "batched vector-matrix matmul restores the rhs VJP shape");

        const std::array<float, 4> broadcast_matmul_lhs_values{1.0F, 2.0F, 3.0F, 4.0F};
        const std::array<float, 6> broadcast_matmul_rhs_values{
            10.0F, 1.0F,
            2.0F, 3.0F,
            -1.0F, 5.0F};
        const Tensor broadcast_matmul_lhs = from_data(
            broadcast_matmul_lhs_values,
            Shape{2, 1, 1, 2});
        const Tensor broadcast_matmul_rhs = from_data(
            broadcast_matmul_rhs_values,
            Shape{3, 2, 1});
        const std::array<Tensor, 2> broadcast_matmul_inputs{
            broadcast_matmul_lhs,
            broadcast_matmul_rhs};
        const std::vector<Tensor> broadcast_matmul_gradients = grad(
            sum(matmul(broadcast_matmul_lhs, broadcast_matmul_rhs)),
            broadcast_matmul_inputs);
        const std::array<float, 4> expected_broadcast_matmul_lhs_gradient{
            11.0F, 9.0F,
            11.0F, 9.0F};
        const std::array<float, 6> expected_broadcast_matmul_rhs_gradient{
            4.0F, 6.0F,
            4.0F, 6.0F,
            4.0F, 6.0F};
        expect(broadcast_matmul_gradients[0].shape() == broadcast_matmul_lhs.shape() &&
                   std::ranges::equal(
                       to_vector(broadcast_matmul_gradients[0]),
                       expected_broadcast_matmul_lhs_gradient),
               "batched matmul reduces the lhs VJP over its broadcast batch axes");
        expect(broadcast_matmul_gradients[1].shape() == broadcast_matmul_rhs.shape() &&
                   std::ranges::equal(
                       to_vector(broadcast_matmul_gradients[1]),
                       expected_broadcast_matmul_rhs_gradient),
               "batched matmul reduces the rhs VJP over its broadcast batch axes");

        const std::array<float, 2> dividend_values{2.0F, 4.0F};
        const std::array<float, 3> divisor_values{1.0F, 2.0F, 4.0F};
        const Tensor dividend = from_data(dividend_values, Shape{2, 1});
        const Tensor divisor = from_data(divisor_values, Shape{1, 3});
        const Tensor quotient_loss = sum(dividend / divisor - dividend);
        const std::array<Tensor, 2> quotient_inputs{dividend, divisor};
        const std::vector<Tensor> quotient_gradients = grad(quotient_loss, quotient_inputs);
        const std::array<float, 2> expected_dividend_gradient{-1.25F, -1.25F};
        const std::array<float, 3> expected_divisor_gradient{-6.0F, -1.5F, -0.375F};
        expect(quotient_gradients[0].shape() == dividend.shape() &&
                   std::ranges::equal(to_vector(quotient_gradients[0]), expected_dividend_gradient),
               "division and subtraction VJPs reduce the dividend cotangent after broadcasting");
        expect(quotient_gradients[1].shape() == divisor.shape() &&
                   std::ranges::equal(to_vector(quotient_gradients[1]), expected_divisor_gradient),
               "division VJP negates and reduces the divisor cotangent after broadcasting");

        const std::array<float, 6> matrix_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        const Tensor matrix = from_data(matrix_values, Shape{2, 3});
        constexpr std::array<Axis, 2> swapped_axes{1, 0};
        const Tensor transformed = permute(reshape(matrix, Shape{3, 2}), swapped_axes);
        const Tensor transform_seed = from_data(matrix_values, Shape{2, 3});
        const std::array<Tensor, 1> matrix_target{matrix};
        const std::vector<Tensor> transformed_vjp = vjp(
            transformed, matrix_target, transform_seed);
        const std::array<float, 6> expected_transformed_vjp{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
        expect(transformed_vjp.size() == 1 &&
                   transformed_vjp.front().shape() == matrix.shape() &&
                   std::ranges::equal(to_vector(transformed_vjp.front()), expected_transformed_vjp),
               "vjp applies inverse permutation and reshape rules to a supplied cotangent");

        const std::vector<Tensor> contiguous_gradients = grad(
            sum(contiguous(transpose(matrix))), matrix_target);
        const std::array<float, 6> expected_contiguous_gradient{
            1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        expect(contiguous_gradients.size() == 1 &&
                   contiguous_gradients.front().shape() == matrix.shape() &&
                   std::ranges::equal(
                       to_vector(contiguous_gradients.front()),
                       expected_contiguous_gradient),
               "contiguous preserves cotangents through a materializing copy");

        const Tensor row = from_data(rhs_values, Shape{1, 3});
        const Tensor broadcasted = broadcast_to(row, Shape{2, 3});
        const Tensor broadcast_seed = from_data(matrix_values, Shape{2, 3});
        const std::array<Tensor, 1> row_target{row};
        const std::vector<Tensor> broadcast_vjp = vjp(
            broadcasted, row_target, broadcast_seed);
        const std::array<float, 3> expected_broadcast_vjp{5.0F, 7.0F, 9.0F};
        expect(broadcast_vjp.size() == 1 &&
                   broadcast_vjp.front().shape() == row.shape() &&
                   std::ranges::equal(to_vector(broadcast_vjp.front()), expected_broadcast_vjp),
               "broadcast_to VJP sums cotangents along expanded dimensions");

        constexpr std::array<Axis, 1> last_axis{1};
        const Tensor kept_sum = sum(matrix, last_axis, true);
        const std::array<float, 2> kept_sum_seed_values{2.0F, 4.0F};
        const Tensor kept_sum_seed = from_data(kept_sum_seed_values, Shape{2, 1});
        const std::vector<Tensor> kept_sum_vjp = vjp(
            kept_sum, matrix_target, kept_sum_seed);
        const std::array<float, 6> expected_kept_sum_vjp{2.0F, 2.0F, 2.0F, 4.0F, 4.0F, 4.0F};
        expect(kept_sum_vjp.size() == 1 &&
                   kept_sum_vjp.front().shape() == matrix.shape() &&
                   std::ranges::equal(to_vector(kept_sum_vjp.front()), expected_kept_sum_vjp),
               "sum VJP broadcasts a kept-dimension cotangent over the reduced axis");

        const std::array<float, 2> unary_values{1.0F, 2.0F};
        const Tensor unary_input = from_data(unary_values, Shape{2});
        const std::array<Tensor, 1> unary_target{unary_input};

        const std::array<float, 2> expected_exp_gradient{std::exp(1.0F), std::exp(2.0F)};
        expect_near(
            to_vector(grad(sum(exp(unary_input)), unary_target).front()),
            expected_exp_gradient,
            1.0E-6F,
            "exp VJP reuses the forward output");

        const std::array<float, 2> expected_log_gradient{1.0F, 0.5F};
        expect_near(
            to_vector(grad(sum(log(unary_input)), unary_target).front()),
            expected_log_gradient,
            1.0E-6F,
            "log VJP divides by its input");

        const std::array<float, 2> expected_sqrt_gradient{
            0.5F, 0.5F / std::sqrt(2.0F)};
        expect_near(
            to_vector(grad(sum(sqrt(unary_input)), unary_target).front()),
            expected_sqrt_gradient,
            1.0E-6F,
            "sqrt VJP scales by twice the forward output");

        const std::array<float, 2> expected_tanh_gradient{
            1.0F - std::tanh(1.0F) * std::tanh(1.0F),
            1.0F - std::tanh(2.0F) * std::tanh(2.0F)};
        expect_near(
            to_vector(grad(sum(tanh(unary_input)), unary_target).front()),
            expected_tanh_gradient,
            1.0E-6F,
            "tanh VJP uses one minus the squared forward output");

        expect_throws<std::invalid_argument>(
            [&transformed, &matrix_target]
            {
                (void)vjp(transformed, matrix_target, full(Shape{1}, 1.0F));
            },
            "public vjp rejects a cotangent with the wrong shape");
        expect_throws<std::invalid_argument>(
            [&transformed, &matrix_target]
            {
                const Tensor wrong_device = full(
                    Shape{2, 3}, 1.0F,
                    TensorOptions{DType::Float32, Device::cpu(1)});
                (void)vjp(transformed, matrix_target, wrong_device);
            },
            "public vjp rejects a cotangent on the wrong device");
        expect_throws<std::invalid_argument>(
            [&transformed, &matrix_target]
            {
                (void)grad(transformed, matrix_target);
            },
            "grad rejects a non-scalar output");
    }
}
