#include <minitensor/data.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/tensor_access.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_reduce_to_shape_test()
    {
        const std::array<float, 6> matrix_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        const Tensor matrix = from_data(matrix_values, Shape{2, 3});

        const Tensor unchanged = detail::reduce_to_shape(matrix, Shape{2, 3});
        expect(detail::TensorAccess::value(unchanged).get() ==
                   detail::TensorAccess::value(matrix).get(),
               "reducing to the same shape reuses the original tensor value");

        const Tensor leading_reduction = detail::reduce_to_shape(matrix, Shape{3});
        const std::array<float, 3> expected_columns{5.0F, 7.0F, 9.0F};
        expect(leading_reduction.shape() == Shape{3} &&
                   std::ranges::equal(to_vector(leading_reduction), expected_columns),
               "reducing an extra leading dimension sums its values and removes the dimension");

        const Tensor aligned_reduction = detail::reduce_to_shape(matrix, Shape{2, 1});
        const std::array<float, 2> expected_rows{6.0F, 15.0F};
        expect(aligned_reduction.shape() == Shape{2, 1} &&
                   std::ranges::equal(to_vector(aligned_reduction), expected_rows),
               "reducing an aligned singleton target dimension retains its position");
        expect(aligned_reduction.dtype() == matrix.dtype() &&
                   aligned_reduction.device() == matrix.device(),
               "reduce_to_shape preserves tensor dtype and device");

        const std::array<float, 12> volume_values{
            1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
            7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
        const Tensor volume = from_data(volume_values, Shape{2, 2, 3});
        const Tensor combined_reduction = detail::reduce_to_shape(volume, Shape{1, 3});
        const std::array<float, 3> expected_combined{22.0F, 26.0F, 30.0F};
        expect(combined_reduction.shape() == Shape{1, 3} &&
                   std::ranges::equal(to_vector(combined_reduction), expected_combined),
               "reducing leading and aligned dimensions together produces the target shape and values");

        const Tensor leading_singleton = from_data(matrix_values, Shape{1, 2, 3});
        const Tensor squeezed = detail::reduce_to_shape(leading_singleton, Shape{2, 3});
        expect(squeezed.shape() == Shape{2, 3} &&
                   std::ranges::equal(to_vector(squeezed), matrix_values),
               "an extra leading singleton dimension is removed without changing values");

        const Tensor scalar = detail::reduce_to_shape(matrix, Shape{});
        expect(scalar.shape().is_scalar() && item(scalar) == 21.0F,
               "reducing to a scalar sums all input dimensions");

        const Tensor empty = from_data(std::span<const float>{}, Shape{0, 3});
        const Tensor empty_reduction = detail::reduce_to_shape(empty, Shape{1, 3});
        const std::array<float, 3> expected_zeroes{};
        expect(empty_reduction.shape() == Shape{1, 3} &&
                   std::ranges::equal(to_vector(empty_reduction), expected_zeroes),
               "reducing an empty dimension produces additive-identity values");

        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                (void)detail::reduce_to_shape(matrix, Shape{1, 2, 3});
            },
            "reduce_to_shape rejects a target with higher rank");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                (void)detail::reduce_to_shape(matrix, Shape{2, 2});
            },
            "reduce_to_shape rejects a target incompatible with broadcasting");
    }
}
