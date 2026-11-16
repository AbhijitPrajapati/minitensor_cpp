#include <minitensor/data.hpp>
#include <minitensor/evaluation.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/types.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
class CudaCopyKernelTest : public CudaDeviceTest
{
  protected:
    template <TensorElement T>
    void expect_copy(Shape shape, detail::Layout input_layout, const std::vector<T> &source,
                     const std::vector<T> &expected)
    {
        CudaKernelHarness harness;
        const detail::TensorSpec spec{shape, dtype_of_v<T>, harness.runtime().device()};

        const detail::BufferRef input_buffer = harness.allocate<T>(source.size());
        harness.copy_from_host<T>(input_buffer, source);
        const detail::Materialization input_storage{input_buffer, input_layout};
        input_storage.validate(spec);
        const detail::TensorView input{spec, input_storage};

        // Guard elements detect writes outside the output's nonzero offset.
        std::vector<T> result(expected.size() + 2, T{-42});
        const detail::BufferRef output_buffer = harness.allocate<T>(result.size());
        harness.copy_from_host<T>(output_buffer, result);
        const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(shape, 1)};
        output_storage.validate(spec);
        const detail::MutableTensorView output{spec, output_storage};
        const detail::ContiguousPrimitive primitive;

        harness.invoke(primitive, std::span<const detail::TensorView>{&input, 1}, output);
        result = harness.copy_to_host<T>(output_buffer);

        std::vector<T> expected_storage{T{-42}};
        expected_storage.insert(expected_storage.end(), expected.begin(), expected.end());
        expected_storage.push_back(T{-42});
        EXPECT_EQ(result, expected_storage);

        EXPECT_EQ(harness.copy_to_host<T>(input_buffer), source);
    }
};

TEST_F(CudaCopyKernelTest, CopiesContiguousInputsWithNonzeroOffsets)
{
    expect_copy<float>(Shape{2, 2}, detail::Layout{{2, 1}, 2}, {90, 91, 1, 2, 3, 4, 99}, {1, 2, 3, 4});
}

TEST_F(CudaCopyKernelTest, CopiesTransposedInputsWithNonzeroOffsets)
{
    expect_copy<float>(Shape{3, 2}, detail::Layout{{1, 3}, 1}, {99, 1, 2, 3, 4, 5, 6, 88}, {1, 4, 2, 5, 3, 6});
}

TEST_F(CudaCopyKernelTest, CopiesNegativeStridesInLogicalOrder)
{
    expect_copy<float>(Shape{2, 3}, detail::Layout{{3, -1}, 3}, {99, 1, 2, 3, 4, 5, 6, 88}, {3, 2, 1, 6, 5, 4});
}

TEST_F(CudaCopyKernelTest, CopiesBroadcastInputs)
{
    expect_copy<float>(Shape{2, 3}, detail::Layout{{0, 1}, 1}, {99, 7, 8, 9}, {7, 8, 9, 7, 8, 9});
}

TEST_F(CudaCopyKernelTest, PreservesInt64ValuesWithoutFloatingPointConversion)
{
    constexpr std::int64_t value = 9'007'199'254'740'993;
    expect_copy<std::int64_t>(Shape{2}, detail::Layout{{2}, 1}, {0, value, 0, -value}, {value, -value});
}

TEST_F(CudaCopyKernelTest, CopiesScalarInputs)
{
    expect_copy<float>(Shape{}, detail::Layout{{}, 1}, {99, 7, 88}, {7});
}

TEST_F(CudaCopyKernelTest, AcceptsEmptyInputsWithoutTouchingOutputStorage)
{
    expect_copy<float>(Shape{2, 0}, detail::Layout{{0, 1}}, {}, {});
}

TEST_F(CudaCopyKernelTest, CopiesElementCountsThatAreNotMultiplesOfTheBlockSize)
{
    std::vector<float> source(2007, -1.0F);
    std::vector<float> expected(1003);
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = static_cast<float>(index);
        source[1 + 2 * index] = expected[index];
    }
    expect_copy<float>(Shape{1003}, detail::Layout{{2}, 1}, source, expected);
}

TEST_F(CudaCopyKernelTest, ReadsBackPublicCudaTensors)
{
    const Device device = Device::cuda();
    const Tensor imported = from_data<float>({1, 2, 3}, Shape{3}, device);
    EXPECT_EQ(to_vector<float>(imported), (std::vector<float>{1, 2, 3}));

    const Tensor filled = full(Shape{2, 3}, 4.0F, TensorOptions{DType::Float32, device});
    eval(filled);
    EXPECT_EQ(to_vector<float>(filled), (std::vector<float>(6, 4.0F)));
}
} // namespace minitensor::test
