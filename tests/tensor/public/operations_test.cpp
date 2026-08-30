#include <minitensor/ops.hpp>

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
    }
}
