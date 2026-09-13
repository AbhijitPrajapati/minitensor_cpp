#include <minitensor/data.hpp>
#include <minitensor/ops.hpp>

#include <algorithm>
#include <array>
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
        expect(std::ranges::equal(
                   to_vector(reshape(permuted_matrix, Shape{2, 3})), expected_permuted),
               "reshape preserves logical element order for a noncontiguous input");

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

        const std::array<float, 2> lhs_values{1.0F, 2.0F};
        const std::array<float, 3> rhs_values{10.0F, 20.0F, 30.0F};
        const Tensor added = from_data(lhs_values, Shape{2, 1}) +
                             from_data(rhs_values, Shape{1, 3});
        const std::array<float, 6> expected_sum{11.0F, 21.0F, 31.0F, 12.0F, 22.0F, 32.0F};
        expect(std::ranges::equal(to_vector(added), expected_sum),
               "to_vector evaluates a graph whose materialized leaves came from host data");

        constexpr std::array<Axis, 1> last_axis{-1};
        const std::array<float, 2> expected_row_sums{6.0F, 15.0F};
        expect(std::ranges::equal(to_vector(sum(matrix, last_axis)), expected_row_sums),
               "sum reduces a selected axis and accepts its negative spelling");
        expect(item(sum(matrix)) == 21.0F,
               "sum without explicit axes reduces all input elements");

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
