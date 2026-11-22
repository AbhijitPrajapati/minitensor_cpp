#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
class CudaElementwiseKernelTest : public CudaDeviceTest
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

    // One representative operation per arity exercises the shared iteration paths.
    void expect_negate(const Operand &input, const std::vector<float> &expected)
    {
        ASSERT_EQ(input.shape.numel(), expected.size());
        CudaKernelHarness harness;
        const detail::TensorSpec spec{input.shape, DType::Float32, harness.runtime().device()};
        const auto source = make_storage(harness, spec, input.layout, input.values);
        const auto destination = make_storage(harness, spec, detail::Layout::contiguous(spec.shape, 1),
                                              std::vector<float>(expected.size() + 2, -42.0F));
        const std::array<detail::TensorView, 1> inputs{detail::TensorView{spec, source}};
        harness.invoke(detail::NegatePrimitive{}, inputs, {spec, destination});

        std::vector<float> expected_storage{-42.0F};
        expected_storage.insert(expected_storage.end(), expected.begin(), expected.end());
        expected_storage.push_back(-42.0F);
        EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), expected_storage);
        EXPECT_EQ(harness.copy_to_host<float>(source.buffer_ref()), input.values);
    }

    void expect_subtract(const Operand &lhs, const Operand &rhs, const Shape &shape, const std::vector<float> &expected)
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
        harness.invoke(detail::SubtractPrimitive{}, inputs, {output_spec, destination});

        std::vector<float> expected_storage{-42.0F};
        expected_storage.insert(expected_storage.end(), expected.begin(), expected.end());
        expected_storage.push_back(-42.0F);
        EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), expected_storage);
        EXPECT_EQ(harness.copy_to_host<float>(lhs_storage.buffer_ref()), lhs.values);
        EXPECT_EQ(harness.copy_to_host<float>(rhs_storage.buffer_ref()), rhs.values);
    }
};

TEST_F(CudaElementwiseKernelTest, UnaryHandlesLinearAndStridedTailsWithOffsets)
{
    std::vector<float> linear(1005, 99.0F);
    std::vector<float> strided(2007, 99.0F);
    std::vector<float> expected(1003);
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto value = static_cast<float>(i + 1);
        linear[i + 1] = value;
        strided[1 + 2 * i] = value;
        expected[i] = -value;
    }
    expect_negate({Shape{1003}, detail::Layout{{1}, 1}, linear}, expected);
    expect_negate({Shape{1003}, detail::Layout{{2}, 1}, strided}, expected);
}

TEST_F(CudaElementwiseKernelTest, UnaryHandlesNegativeTransposedAndBroadcastLayouts)
{
    expect_negate({Shape{3, 2}, detail::Layout{{-1, 3}, 3}, {99, 1, 2, 3, 4, 5, 6, 88}}, {-3, -6, -2, -5, -1, -4});
    expect_negate({Shape{2, 3}, detail::Layout{{0, 0}, 1}, {99, 5, 88}}, {-5, -5, -5, -5, -5, -5});
}

TEST_F(CudaElementwiseKernelTest, BinaryHandlesLinearAndStridedTailsWithOffsetsAndOperandOrder)
{
    std::vector<float> lhs(1005, 99.0F);
    std::vector<float> strided_lhs(2007, 99.0F);
    std::vector<float> rhs(1006, 88.0F);
    std::vector<float> expected(1003);
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        const auto value = static_cast<float>(i);
        lhs[i + 1] = 2.0F * value + 10.0F;
        strided_lhs[1 + 2 * i] = lhs[i + 1];
        rhs[i + 2] = value;
        expected[i] = value + 10.0F;
    }
    expect_subtract({Shape{1003}, detail::Layout{{1}, 1}, lhs}, {Shape{1003}, detail::Layout{{1}, 2}, rhs}, Shape{1003},
                    expected);
    expect_subtract({Shape{1003}, detail::Layout{{2}, 1}, strided_lhs}, {Shape{1003}, detail::Layout{{1}, 2}, rhs},
                    Shape{1003}, expected);
}

TEST_F(CudaElementwiseKernelTest, BinaryLinearSupportsEitherScalarOperand)
{
    const Operand scalar{Shape{1, 1}, detail::Layout{{0, 0}, 1}, {99, 10, 88}};
    const Operand tensor{Shape{2, 3}, detail::Layout{{3, 1}, 1}, {99, 1, 2, 3, 4, 5, 6, 88}};
    expect_subtract(scalar, tensor, Shape{2, 3}, {9, 8, 7, 6, 5, 4});
    expect_subtract(tensor, scalar, Shape{2, 3}, {-9, -8, -7, -6, -5, -4});
}

TEST_F(CudaElementwiseKernelTest, BinaryStridedHandlesNegativeStridesAndRankExpansion)
{
    expect_subtract({Shape{2, 3}, detail::Layout{{-3, 1}, 4}, {99, 1, 2, 3, 4, 5, 6, 88}},
                    {Shape{3}, detail::Layout{{-1}, 3}, {99, 10, 20, 30, 88}}, Shape{2, 3},
                    {-26, -15, -4, -29, -18, -7});
}

TEST_F(CudaElementwiseKernelTest, BinaryStridedBroadcastsBothOperands)
{
    expect_subtract({Shape{2, 1}, detail::Layout{{1, 1}, 1}, {99, 10, 20, 88}},
                    {Shape{1, 3}, detail::Layout{{3, -1}, 3}, {99, 1, 2, 3, 88}}, Shape{2, 3}, {7, 8, 9, 17, 18, 19});
}

TEST_F(CudaElementwiseKernelTest, HandlesScalarAndEmptyOutputs)
{
    expect_negate({Shape{}, detail::Layout{{}, 1}, {99, 7, 88}}, {-7});
    expect_subtract({Shape{}, detail::Layout{{}, 1}, {99, 7, 88}},
                    {Shape{1, 1}, detail::Layout{{0, 0}, 1}, {99, 12, 88}}, Shape{1, 1}, {-5});
    expect_negate({Shape{2, 0}, detail::Layout::contiguous(Shape{2, 0}), {}}, {});
    expect_subtract({Shape{2, 0}, detail::Layout::contiguous(Shape{2, 0}), {}},
                    {Shape{1, 0}, detail::Layout::contiguous(Shape{1, 0}), {}}, Shape{2, 0}, {});
}
} // namespace minitensor::test
