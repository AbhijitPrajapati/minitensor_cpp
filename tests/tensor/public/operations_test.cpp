#include <minitensor/ops.hpp>

#include <array>
#include <span>
#include <stdexcept>
#include <utility>

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_operations_test()
    {
        const Tensor matrix = full(Shape{2, 3}, 4.25F);
        expect(matrix.shape() == Shape{2, 3}, "full exposes its requested shape");
        expect(matrix.rank() == 2, "a tensor handle reports its rank");
        expect(matrix.numel() == 6, "a tensor handle reports its element count");
        expect(matrix.dtype() == DType::Float32, "full uses the default dtype");
        expect(matrix.device() == Device::cpu(), "full uses the default device");

        const Tensor scalar = full(Shape{}, -1.0F);
        expect(scalar.shape().is_scalar(), "full constructs scalar tensors");
        expect(scalar.rank() == 0, "a scalar tensor handle has rank zero");
        expect(scalar.numel() == 1, "a scalar tensor handle has one element");

        const Tensor empty = full(Shape{2, 0, 3}, 0.0F);
        expect(empty.numel() == 0, "full preserves empty tensor shapes");

        const TensorOptions options{DType::Float32, Device::cpu(7)};
        const Tensor configured = full(Shape{3}, 2.0F, options);
        expect(configured.dtype() == options.dtype, "full preserves an explicit dtype option");
        expect(configured.device() == options.device, "full preserves an explicit device option");

        Tensor copied = matrix;
        expect(copied.shape() == matrix.shape(), "copy construction preserves tensor metadata");
        Tensor assigned = full(Shape{1}, 0.0F);
        assigned = matrix;
        expect(assigned.shape() == matrix.shape(), "copy assignment replaces the tensor handle");
        const Tensor moved{std::move(copied)};
        expect(moved.shape() == matrix.shape(), "move construction transfers the tensor handle");

        const Tensor same_shape_sum = matrix + full(Shape{2, 3}, 1.0F);
        expect(same_shape_sum.shape() == Shape{2, 3}, "addition preserves equal input shapes");
        expect(same_shape_sum.dtype() == matrix.dtype(), "addition preserves the input dtype");
        expect(same_shape_sum.device() == matrix.device(), "addition preserves the input device");

        const Tensor broadcast_sum = full(Shape{2, 1, 4}, 1.0F) + full(Shape{3, 4}, 2.0F);
        expect(broadcast_sum.shape() == Shape{2, 3, 4},
               "addition broadcasts singleton and missing leading dimensions");

        const Tensor scalar_sum = scalar + full(Shape{2, 3}, 1.0F);
        expect(scalar_sum.shape() == Shape{2, 3}, "addition broadcasts a scalar to a ranked tensor");

        const Tensor empty_sum = full(Shape{2, 0, 3}, 1.0F) + full(Shape{1, 3}, 2.0F);
        expect(empty_sum.shape() == Shape{2, 0, 3}, "addition broadcasts compatible empty shapes");
        expect(empty_sum.numel() == 0, "a broadcasted empty result remains empty");

        constexpr std::array<Axis, 2> swapped_axes{1, 0};
        const Tensor permuted_matrix = permute(matrix, swapped_axes);
        expect(permuted_matrix.shape() == Shape{3, 2},
               "permutation reorders tensor dimensions");
        expect(permuted_matrix.dtype() == matrix.dtype() &&
                   permuted_matrix.device() == matrix.device(),
               "permutation preserves tensor dtype and device");

        constexpr std::array<Axis, 2> negative_swapped_axes{-1, -2};
        expect(permute(matrix, negative_swapped_axes).shape() == Shape{3, 2},
               "permutation accepts normalized negative axes");
        expect(permute(scalar, std::span<const Axis>{}).shape().is_scalar(),
               "an empty permutation preserves a scalar tensor");

        const Tensor reshaped_matrix = reshape(matrix, Shape{3, 2});
        expect(reshaped_matrix.shape() == Shape{3, 2},
               "reshape replaces the tensor shape while preserving its element count");
        expect(reshaped_matrix.dtype() == matrix.dtype() &&
                   reshaped_matrix.device() == matrix.device(),
               "reshape preserves tensor dtype and device");
        expect(reshape(scalar, Shape{1}).shape() == Shape{1},
               "reshape can convert a scalar into a ranked singleton tensor");
        expect(reshape(empty, Shape{0, 6}).shape() == Shape{0, 6},
               "reshape accepts a different empty shape");

        const Tensor broadcast_source = full(Shape{2, 1}, 3.0F);
        const Tensor broadcasted_tensor = broadcast_to(broadcast_source, Shape{2, 3});
        expect(broadcasted_tensor.shape() == Shape{2, 3},
               "broadcast_to expands singleton dimensions to the requested shape");
        expect(broadcasted_tensor.dtype() == broadcast_source.dtype() &&
                   broadcasted_tensor.device() == broadcast_source.device(),
               "broadcast_to preserves tensor dtype and device");
        expect(broadcast_to(scalar, Shape{2, 3}).shape() == Shape{2, 3},
               "broadcast_to expands a scalar to a ranked shape");
        expect(broadcast_to(empty, Shape{4, 2, 0, 3}).shape() == Shape{4, 2, 0, 3},
               "broadcast_to supports compatible empty shapes and leading dimensions");

        constexpr std::array<Axis, 1> last_axis{1};
        const Tensor row_sums = sum(matrix, last_axis);
        expect(row_sums.shape() == Shape{2},
               "sum removes a reduced axis from the output shape");
        expect(row_sums.dtype() == matrix.dtype() && row_sums.device() == matrix.device(),
               "sum preserves tensor dtype and device");

        constexpr std::array<Axis, 2> unsorted_axes{-1, 0};
        expect(sum(full(Shape{2, 3, 4}, 1.0F), unsorted_axes).shape() == Shape{3},
               "sum normalizes negative axes and accepts them in any order");
        expect(sum(matrix, std::span<const Axis>{}).shape() == matrix.shape(),
               "sum over no axes preserves the input shape");
        expect(sum(matrix).shape().is_scalar(),
               "sum without explicit axes reduces every matrix axis");
        expect(sum(scalar).shape().is_scalar(),
               "sum without explicit axes preserves a scalar shape");

        constexpr std::array<Axis, 1> empty_reduction_axis{1};
        expect(sum(empty, empty_reduction_axis).shape() == Shape{2, 3},
               "sum removes a zero-length reduction axis while preserving the other axes");

        expect_throws<std::invalid_argument>(
            []
            {
                (void)(full(Shape{2, 3}, 1.0F) + full(Shape{2, 2}, 1.0F));
            },
            "addition rejects incompatible shapes");
        expect_throws<std::invalid_argument>(
            []
            {
                const Tensor lhs = full(Shape{2}, 1.0F, TensorOptions{DType::Float32, Device::cpu(0)});
                const Tensor rhs = full(Shape{2}, 1.0F, TensorOptions{DType::Float32, Device::cpu(1)});
                (void)(lhs + rhs);
            },
            "addition rejects tensors on different devices");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                constexpr std::array<Axis, 1> wrong_size{0};
                (void)permute(matrix, wrong_size);
            },
            "permutation rejects an axis count that differs from the input rank");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                constexpr std::array<Axis, 2> duplicate_axes{0, 0};
                (void)permute(matrix, duplicate_axes);
            },
            "permutation rejects duplicate axes");
        expect_throws<std::out_of_range>(
            [&matrix]
            {
                constexpr std::array<Axis, 2> out_of_range_axes{0, 2};
                (void)permute(matrix, out_of_range_axes);
            },
            "permutation rejects an axis outside the input rank");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                (void)reshape(matrix, Shape{5});
            },
            "reshape rejects a shape with a different element count");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                (void)broadcast_to(matrix, Shape{2, 2});
            },
            "broadcast_to rejects incompatible dimensions");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                (void)broadcast_to(matrix, Shape{3});
            },
            "broadcast_to rejects a target with fewer dimensions");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                constexpr std::array<Axis, 2> duplicate_axes{1, -1};
                (void)sum(matrix, duplicate_axes);
            },
            "sum rejects axes that become duplicates after normalization");
        expect_throws<std::out_of_range>(
            [&matrix]
            {
                constexpr std::array<Axis, 1> out_of_range_axis{2};
                (void)sum(matrix, out_of_range_axis);
            },
            "sum rejects an axis outside the input rank");
        expect_throws<std::invalid_argument>(
            [&matrix]
            {
                constexpr std::array<Axis, 3> too_many_axes{0, 1, 0};
                (void)sum(matrix, too_many_axes);
            },
            "sum rejects more reduction axes than the input rank");
    }
}
