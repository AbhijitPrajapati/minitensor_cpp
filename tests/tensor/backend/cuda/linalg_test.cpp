#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
class CudaLinalgKernelTest : public CudaDeviceTest
{
  protected:
    struct Operand
    {
        Shape shape;
        detail::Layout layout;
        std::vector<float> values;
    };

    detail::Materialization make_storage(CudaKernelHarness &harness, const detail::TensorSpec &spec,
                                         detail::Layout layout, const std::vector<float> &values)
    {
        auto buffer = harness.allocate<float>(values.size());
        harness.copy_from_host<float>(buffer, values);
        detail::Materialization storage{std::move(buffer), std::move(layout)};
        storage.validate(spec);
        return storage;
    }

    void expect_matmul(const Operand &lhs, const Operand &rhs, const Shape &shape, const std::vector<float> &expected)
    {
        ASSERT_EQ(shape.numel(), expected.size());
        CudaKernelHarness harness;
        const auto device = harness.runtime().device();
        const detail::TensorSpec lhs_spec{lhs.shape, DType::Float32, device};
        const detail::TensorSpec rhs_spec{rhs.shape, DType::Float32, device};
        const detail::TensorSpec output_spec{shape, DType::Float32, device};
        const auto lhs_storage = make_storage(harness, lhs_spec, lhs.layout, lhs.values);
        const auto rhs_storage = make_storage(harness, rhs_spec, rhs.layout, rhs.values);
        const auto destination = make_storage(harness, output_spec, detail::Layout::contiguous(shape, 1),
                                              std::vector<float>(expected.size() + 2, -42.0F));
        const std::array<detail::TensorView, 2> inputs{detail::TensorView{lhs_spec, lhs_storage},
                                                       detail::TensorView{rhs_spec, rhs_storage}};
        harness.invoke(detail::MatmulPrimitive{}, inputs, {output_spec, destination});

        const auto result = harness.copy_to_host<float>(destination.buffer_ref());
        ASSERT_EQ(result.size(), expected.size() + 2);
        EXPECT_EQ(result.front(), -42.0F);
        EXPECT_EQ(result.back(), -42.0F);
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            SCOPED_TRACE(i);
            EXPECT_FLOAT_EQ(result[i + 1], expected[i]);
        }
        EXPECT_EQ(harness.copy_to_host<float>(lhs_storage.buffer_ref()), lhs.values);
        EXPECT_EQ(harness.copy_to_host<float>(rhs_storage.buffer_ref()), rhs.values);
    }
};

TEST_F(CudaLinalgKernelTest, MatmulHonorsInputAndOutputOffsets)
{
    expect_matmul({Shape{2, 3}, detail::Layout{{3, 1}, 1}, {99, 1, 2, 3, 4, 5, 6, 88}},
                  {Shape{3, 2}, detail::Layout{{2, 1}, 2}, {99, 99, 7, 8, 9, 10, 11, 12, 88}}, Shape{2, 2},
                  {58, 64, 139, 154});
}

TEST_F(CudaLinalgKernelTest, DotHandlesNegativeAndBroadcastStrides)
{
    expect_matmul({Shape{3}, detail::Layout{{-1}, 3}, {99, 1, 2, 3, 88}},
                  {Shape{3}, detail::Layout{{0}, 1}, {99, 2, 88}}, Shape{}, {12});
}

TEST_F(CudaLinalgKernelTest, MatrixVectorHandlesReversedBatchesAndStridedVectors)
{
    expect_matmul({Shape{2, 2, 3}, detail::Layout{{-6, 3, 1}, 7}, {99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 88}},
                  {Shape{3}, detail::Layout{{2}, 1}, {99, 1, 99, 2, 99, 3, 88}}, Shape{2, 2}, {50, 68, 14, 32});
}

TEST_F(CudaLinalgKernelTest, VectorMatrixHandlesBatchedTransposedMatrices)
{
    expect_matmul({Shape{3}, detail::Layout{{1}, 1}, {99, 1, 2, 3, 88}},
                  {Shape{2, 3, 2}, detail::Layout{{6, 1, 3}, 1}, {99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 88}},
                  Shape{2, 2}, {14, 32, 50, 68});
}

TEST_F(CudaLinalgKernelTest, MatrixMatrixHandlesTransposedAndReversedMatrixAxes)
{
    expect_matmul({Shape{2, 3}, detail::Layout{{1, -2}, 5}, {99, 1, 4, 2, 5, 3, 6, 88}},
                  {Shape{3, 2}, detail::Layout{{2, 1}, 1}, {99, 7, 8, 9, 10, 11, 12, 88}}, Shape{2, 2},
                  {50, 56, 131, 146});
}

TEST_F(CudaLinalgKernelTest, BroadcastsBothOperandsAcrossMultipleBatchAxes)
{
    expect_matmul({Shape{2, 1, 2, 2}, detail::Layout::contiguous(Shape{2, 1, 2, 2}), {1, 0, 0, 1, 2, 0, 0, 3}},
                  {Shape{3, 2, 2}, detail::Layout::contiguous(Shape{3, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                  Shape{2, 3, 2, 2},
                  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 4, 9, 12, 10, 12, 21, 24, 18, 20, 33, 36});
}

TEST_F(CudaLinalgKernelTest, WritesMultipleBlocksAndPartialFinalBlocks)
{
    std::vector<float> rhs(3 * 19);
    for (std::size_t k = 0; k < 3; ++k)
    {
        for (std::size_t column = 0; column < 19; ++column)
        {
            rhs[k * 19 + column] = static_cast<float>(column + 1);
        }
    }
    std::vector<float> expected(17 * 19);
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = 3.0F * static_cast<float>(i % 19 + 1);
    }
    expect_matmul({Shape{17, 3}, detail::Layout::contiguous(Shape{17, 3}), std::vector<float>(17 * 3, 1.0F)},
                  {Shape{3, 19}, detail::Layout::contiguous(Shape{3, 19}), rhs}, Shape{17, 19}, expected);
}

TEST_F(CudaLinalgKernelTest, ZeroContractionsFillNonemptyOutputsIncludingEmptyDot)
{
    expect_matmul({Shape{0}, detail::Layout{{1}}, {}}, {Shape{0}, detail::Layout{{1}}, {}}, Shape{}, {0});
    expect_matmul({Shape{2, 0}, detail::Layout::contiguous(Shape{2, 0}), {}},
                  {Shape{0, 3}, detail::Layout::contiguous(Shape{0, 3}), {}}, Shape{2, 3}, std::vector<float>(6, 0));
    expect_matmul({Shape{2, 2, 0}, detail::Layout::contiguous(Shape{2, 2, 0}), {}},
                  {Shape{1, 0, 3}, detail::Layout::contiguous(Shape{1, 0, 3}), {}}, Shape{2, 2, 3},
                  std::vector<float>(12, 0));
    expect_matmul({Shape{0}, detail::Layout{{1}}, {}}, {Shape{2, 0, 3}, detail::Layout::contiguous(Shape{2, 0, 3}), {}},
                  Shape{2, 3}, std::vector<float>(6, 0));
    expect_matmul({Shape{2, 2, 0}, detail::Layout::contiguous(Shape{2, 2, 0}), {}}, {Shape{0}, detail::Layout{{1}}, {}},
                  Shape{2, 2}, std::vector<float>(4, 0));
}

TEST_F(CudaLinalgKernelTest, EmptyOutputsLeaveStorageUntouched)
{
    expect_matmul({Shape{0, 3}, detail::Layout::contiguous(Shape{0, 3}), {}},
                  {Shape{3, 2}, detail::Layout::contiguous(Shape{3, 2}), std::vector<float>(6, 1)}, Shape{0, 2}, {});
    expect_matmul({Shape{3}, detail::Layout{{1}}, {1, 2, 3}},
                  {Shape{3, 0}, detail::Layout::contiguous(Shape{3, 0}), {}}, Shape{0}, {});
    expect_matmul({Shape{0, 3}, detail::Layout::contiguous(Shape{0, 3}), {}},
                  {Shape{3}, detail::Layout{{1}}, {1, 2, 3}}, Shape{0}, {});
    expect_matmul({Shape{0, 2, 3}, detail::Layout::contiguous(Shape{0, 2, 3}), {}},
                  {Shape{1, 3, 4}, detail::Layout::contiguous(Shape{1, 3, 4}), std::vector<float>(12, 1)},
                  Shape{0, 2, 4}, {});
}
} // namespace minitensor::test
