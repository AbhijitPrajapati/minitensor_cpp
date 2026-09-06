#include <minitensor/evaluation.hpp>
#include <minitensor/ops.hpp>

#include <array>
#include <span>
#include <stdexcept>

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_evaluation_test()
    {
        eval(std::span<const Tensor>{});

        const Tensor scalar = full(Shape{}, 3.5F);
        eval(scalar);
        eval(scalar);
        expect(scalar.shape().is_scalar(),
               "single-tensor evaluation preserves a cached tensor handle");

        const Tensor lhs = full(Shape{2, 1}, 1.5F);
        const Tensor rhs = full(Shape{1, 3}, 2.5F);
        const Tensor sum = lhs + rhs;
        const Tensor result = sum + full(Shape{2, 3}, 4.0F);
        const std::array<Tensor, 3> roots{lhs, sum, result};
        eval(roots);
        expect(result.shape() == Shape{2, 3},
               "multi-root evaluation executes a shared broadcasted computation graph");

        eval(result);
        expect(result.numel() == 6,
               "re-evaluating a materialized public tensor reuses its cached result");

        const Tensor empty = full(Shape{2, 0, 3}, 7.0F);
        eval(empty);
        expect(empty.numel() == 0, "evaluation supports tensors with zero-byte storage");

        expect_throws<std::runtime_error>(
            []
            {
                const Tensor unsupported = full(
                    Shape{1}, 1.0F, TensorOptions{DType::Float32, Device::cpu(1)});
                eval(unsupported);
            },
            "public evaluation reports a device with no registered runtime");
    }
}
