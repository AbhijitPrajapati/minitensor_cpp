#include <minitensor/types.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "tensor/core/axis.hpp"
#include "tensor/core/broadcast_shape.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_core_test()
    {
        using detail::TensorSpec;

        expect(detail::normalize_axis(Axis{1}, 3) == 1,
               "axis normalization preserves a nonnegative axis");
        expect(detail::normalize_axis(Axis{-1}, 3) == 2,
               "axis normalization resolves a negative axis from the end");
        expect_throws<std::out_of_range>(
            []
            {
                (void)detail::normalize_axis(Axis{3}, 3);
            },
            "axis normalization rejects an axis at the positive rank boundary");
        expect_throws<std::out_of_range>(
            []
            {
                (void)detail::normalize_axis(Axis{-4}, 3);
            },
            "axis normalization rejects an axis before the negative rank boundary");

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

        constexpr auto overflowing_extent = static_cast<Extent>(
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
