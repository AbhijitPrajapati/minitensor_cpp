#include <minitensor/types.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "tensor/core/broadcast_shape.hpp"
#include "tensor/core/checked_arithmetic.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_core_test()
    {
        using detail::TensorSpec;
        using detail::checked_add;
        using detail::checked_multiply;

        expect(checked_multiply(std::int64_t{-6}, std::int64_t{7}) == -42,
               "checked multiplication returns a representable signed product");
        expect(checked_add(std::int64_t{-6}, std::int64_t{7}) == 1,
               "checked addition returns a representable signed sum");
        expect(checked_multiply(std::size_t{6}, std::size_t{7}) == 42 &&
                   checked_add(std::size_t{6}, std::size_t{7}) == 13,
               "checked arithmetic supports unsigned integers");

        expect_throws<std::overflow_error>(
            []
            {
                (void)checked_multiply(
                    std::numeric_limits<std::int64_t>::max(), std::int64_t{2});
            },
            "checked multiplication rejects signed overflow");
        expect_throws<std::overflow_error>(
            []
            {
                (void)checked_multiply(
                    std::numeric_limits<std::int64_t>::min(), std::int64_t{-1});
            },
            "checked multiplication rejects the minimum signed value times negative one");
        expect_throws<std::overflow_error>(
            []
            {
                (void)checked_add(
                    std::numeric_limits<std::int64_t>::max(), std::int64_t{1});
            },
            "checked addition rejects signed overflow");
        expect_throws<std::overflow_error>(
            []
            {
                (void)checked_add(
                    std::numeric_limits<std::int64_t>::min(), std::int64_t{-1});
            },
            "checked addition rejects signed underflow");
        expect_throws<std::overflow_error>(
            []
            {
                (void)checked_multiply(
                    std::numeric_limits<std::size_t>::max(), std::size_t{2});
            },
            "checked multiplication rejects unsigned overflow");
        expect_throws<std::overflow_error>(
            []
            {
                (void)checked_add(
                    std::numeric_limits<std::size_t>::max(), std::size_t{1});
            },
            "checked addition rejects unsigned overflow");

        expect(detail::broadcast_shape(Shape{2, 3}, Shape{2, 3}) == Shape{2, 3},
               "broadcasting preserves equal shapes");
        expect(detail::broadcast_shape(Shape{2, 1, 4}, Shape{3, 4}) == Shape{2, 3, 4},
               "broadcasting expands singleton dimensions and missing leading dimensions");
        expect(detail::broadcast_shape(Shape{}, Shape{2, 3}) == Shape{2, 3},
               "broadcasting expands a scalar to a ranked shape");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)detail::broadcast_shape(Shape{2, 3}, Shape{2, 2});
            },
            "broadcasting rejects incompatible shapes");

        const TensorSpec scalar_spec{Shape{}, DType::Float32, Device::cpu()};
        expect(detail::dense_size_bytes(scalar_spec) == sizeof(float),
               "dense storage for a scalar contains one element");

        const TensorSpec matrix_spec{Shape{2, 3}, DType::Float32, Device::cpu(4)};
        expect(detail::dense_size_bytes(matrix_spec) == 6 * sizeof(float),
               "dense storage size is the product of element count and dtype size");

        const TensorSpec empty_spec{Shape{2, 0, 3}, DType::Float32, Device::cpu()};
        expect(detail::dense_size_bytes(empty_spec) == 0,
               "an empty tensor requires no dense storage regardless of its other extents");

        const auto overflowing_extent = static_cast<Extent>(
            std::numeric_limits<std::size_t>::max() / sizeof(float) + 1);
        expect_throws<std::overflow_error>(
            [overflowing_extent]
            {
                const TensorSpec oversized{Shape{overflowing_extent}, DType::Float32, Device::cpu()};
                (void)detail::dense_size_bytes(oversized);
            },
            "dense storage size rejects byte-count overflow");
    }
}
