#include <minitensor/types.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_dense_size_test()
    {
        using detail::TensorSpec;

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
