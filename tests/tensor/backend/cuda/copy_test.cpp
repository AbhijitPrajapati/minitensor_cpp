#include <minitensor/data.hpp>
#include <minitensor/evaluation.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/support/device_test.hpp"
#include "tensor/backend/cuda/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/gather.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
class CudaCopyKernelTest : public CudaDeviceTest
{
  protected:
    template <TensorElement T>
    detail::Materialization make_storage(CudaKernelHarness &harness, const detail::TensorSpec &spec,
                                         detail::Layout layout, const std::vector<T> &values)
    {
        auto buffer = harness.allocate<T>(values.size());
        harness.copy_from_host<T>(buffer, values);
        detail::Materialization storage{std::move(buffer), std::move(layout)};
        storage.validate(spec);
        return storage;
    }

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

TEST_F(CudaCopyKernelTest, ReshapeCopiesNoncontiguousInputsInLogicalOrder)
{
    CudaKernelHarness harness;
    const detail::TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{3, 2}, DType::Float32, Device::cuda()};
    const auto source =
        make_storage<float>(harness, input_spec, detail::Layout{{-3, 1}, 4}, {99, 1, 2, 3, 4, 5, 6, 88});
    const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1),
                                                 std::vector<float>(8, -42));
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, source}};
    harness.invoke(detail::ReshapePrimitive{output_spec.shape}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 4, 5, 6, 1, 2, 3, -42}));
}

TEST_F(CudaCopyKernelTest, ConcatenatesStridedAndEmptySections)
{
    CudaKernelHarness harness;
    const detail::TensorSpec column_spec{Shape{2, 1}, DType::Float32, Device::cuda()};
    const detail::TensorSpec empty_spec{Shape{2, 0}, DType::Float32, Device::cuda()};
    const detail::TensorSpec block_spec{Shape{2, 2}, DType::Float32, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 3}, DType::Float32, Device::cuda()};
    const auto column = make_storage<float>(harness, column_spec, detail::Layout{{1, 1}, 1}, {99, 1, 4, 88});
    const auto empty = make_storage<float>(harness, empty_spec, detail::Layout::contiguous(empty_spec.shape), {});
    const auto block = make_storage<float>(harness, block_spec, detail::Layout{{1, 2}, 1}, {99, 2, 5, 3, 6, 88});
    const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1),
                                                 std::vector<float>(8, -42));
    const std::array<detail::TensorView, 3> inputs{detail::TensorView{column_spec, column},
                                                   detail::TensorView{empty_spec, empty},
                                                   detail::TensorView{block_spec, block}};
    harness.invoke(detail::ConcatenatePrimitive{1, 2}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 1, 2, 3, 4, 5, 6, -42}));
    EXPECT_EQ(harness.copy_to_host<float>(block.buffer_ref()), (std::vector<float>{99, 2, 5, 3, 6, 88}));
}

TEST_F(CudaCopyKernelTest, ConcatenatesContiguousInt64Sections)
{
    CudaKernelHarness harness;
    constexpr std::int64_t value = 9'007'199'254'740'993;
    const detail::TensorSpec input_spec{Shape{1, 2}, DType::Int64, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 2}, DType::Int64, Device::cuda()};
    const auto first =
        make_storage<std::int64_t>(harness, input_spec, detail::Layout{{2, 1}, 1}, {99, value, -value, 88});
    const auto second = make_storage<std::int64_t>(harness, input_spec, detail::Layout{{2, 1}, 1}, {99, 7, 8, 88});
    const auto destination = make_storage<std::int64_t>(
        harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1), std::vector<std::int64_t>(6, -42));
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{input_spec, first},
                                                   detail::TensorView{input_spec, second}};
    harness.invoke(detail::ConcatenatePrimitive{0, 2}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<std::int64_t>(destination.buffer_ref()),
              (std::vector<std::int64_t>{-42, value, -value, 7, 8, -42}));
}

TEST_F(CudaCopyKernelTest, ConcatenatesEmptyOutputsWithoutTouchingStorage)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{2, 0}, DType::Float32, Device::cuda()};
    const auto source = make_storage<float>(harness, spec, detail::Layout::contiguous(spec.shape), {});
    const auto destination = make_storage<float>(harness, spec, detail::Layout::contiguous(spec.shape, 1), {-42, -42});
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{spec, source}, detail::TensorView{spec, source}};
    harness.invoke(detail::ConcatenatePrimitive{1, 2}, inputs, {spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, -42}));
}

TEST_F(CudaCopyKernelTest, BasicIndexScatterCopiesStridedUpdatesIntoNegativeStepSlices)
{
    CudaKernelHarness harness;
    const detail::TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 5}, DType::Float32, Device::cuda()};
    const auto source = make_storage<float>(harness, input_spec, detail::Layout{{1, 2}, 1}, {99, 1, 4, 2, 5, 3, 6, 88});
    const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1),
                                                 std::vector<float>(12, -42));
    const auto parameters =
        detail::BasicIndexParameters::for_slice(1, std::nullopt, std::nullopt, -2, output_spec.shape);
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, source}};
    harness.invoke(detail::BasicIndexScatterPrimitive{parameters}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()),
              (std::vector<float>{-42, 3, 0, 2, 0, 1, 6, 0, 5, 0, 4, -42}));
}

TEST_F(CudaCopyKernelTest, BasicIndexScatterPreservesInt64ValuesInRankReducedSelections)
{
    CudaKernelHarness harness;
    constexpr std::int64_t value = 9'007'199'254'740'993;
    const detail::TensorSpec input_spec{Shape{2}, DType::Int64, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 2}, DType::Int64, Device::cuda()};
    const auto source =
        make_storage<std::int64_t>(harness, input_spec, detail::Layout{{-1}, 2}, {99, -value, value, 88});
    const auto destination = make_storage<std::int64_t>(
        harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1), std::vector<std::int64_t>(6, -42));
    const auto parameters = detail::BasicIndexParameters::for_select(0, 1, output_spec.shape);
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, source}};
    harness.invoke(detail::BasicIndexScatterPrimitive{parameters}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<std::int64_t>(destination.buffer_ref()),
              (std::vector<std::int64_t>{-42, 0, 0, value, -value, -42}));
}

TEST_F(CudaCopyKernelTest, BasicIndexScatterHandlesEmptyUpdatesAndOutputs)
{
    CudaKernelHarness harness;
    for (const Shape &shape : {Shape{2, 3}, Shape{0, 3}})
    {
        SCOPED_TRACE(shape.numel());
        const auto parameters = detail::BasicIndexParameters::for_slice(1, 0, 0, 1, shape);
        const detail::TensorSpec input_spec{parameters.output_shape(), DType::Float32, Device::cuda()};
        const detail::TensorSpec output_spec{shape, DType::Float32, Device::cuda()};
        const auto source = make_storage<float>(harness, input_spec, detail::Layout::contiguous(input_spec.shape), {});
        const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(shape, 1),
                                                     std::vector<float>(shape.numel() + 2, -42));
        const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, source}};
        harness.invoke(detail::BasicIndexScatterPrimitive{parameters}, inputs, {output_spec, destination});
        std::vector<float> expected(shape.numel() + 2, 0);
        expected.front() = expected.back() = -42;
        EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), expected);
    }
}

TEST_F(CudaCopyKernelTest, GatherHandlesStridedSourcesIndicesAndOutputOffsets)
{
    CudaKernelHarness harness;
    const detail::TensorSpec source_spec{Shape{2, 3}, DType::Float32, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{2, 2}, DType::Int64, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 2}, DType::Float32, Device::cuda()};
    const auto source =
        make_storage<float>(harness, source_spec, detail::Layout{{-3, 1}, 4}, {99, 1, 2, 3, 4, 5, 6, 88});
    const auto indices =
        make_storage<std::int64_t>(harness, indices_spec, detail::Layout{{1, 2}, 1}, {99, 2, 1, 0, 1, 88});
    const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1),
                                                 std::vector<float>(6, -42));
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{source_spec, source},
                                                   detail::TensorView{indices_spec, indices}};
    harness.invoke(detail::GatherPrimitive{1, 2}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 6, 4, 2, 2, -42}));
    EXPECT_EQ(harness.copy_to_host<float>(source.buffer_ref()), (std::vector<float>{99, 1, 2, 3, 4, 5, 6, 88}));
    EXPECT_EQ(harness.copy_to_host<std::int64_t>(indices.buffer_ref()),
              (std::vector<std::int64_t>{99, 2, 1, 0, 1, 88}));
}

TEST_F(CudaCopyKernelTest, GatherPreservesInt64WithBroadcastIndicesAlongTheFirstAxis)
{
    CudaKernelHarness harness;
    constexpr std::int64_t value = 9'007'199'254'740'993;
    const detail::TensorSpec source_spec{Shape{2, 3}, DType::Int64, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{2, 2}, DType::Int64, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 2}, DType::Int64, Device::cuda()};
    const auto source = make_storage<std::int64_t>(
        harness, source_spec, detail::Layout::contiguous(source_spec.shape, 1), {99, value, -value, 7, 9, 10, 11, 88});
    const auto indices = make_storage<std::int64_t>(harness, indices_spec, detail::Layout{{0, 1}, 1}, {99, 1, 0, 88});
    const auto destination = make_storage<std::int64_t>(
        harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1), std::vector<std::int64_t>(6, -42));
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{source_spec, source},
                                                   detail::TensorView{indices_spec, indices}};
    harness.invoke(detail::GatherPrimitive{0, 2}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<std::int64_t>(destination.buffer_ref()),
              (std::vector<std::int64_t>{-42, 9, -value, 9, -value, -42}));
}

TEST_F(CudaCopyKernelTest, GatherScatterAddHandlesStridedUpdatesAndRepeatedIndices)
{
    CudaKernelHarness harness;
    const detail::TensorSpec updates_spec{Shape{2, 3}, DType::Float32, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{2, 3}, DType::Int64, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{2, 4}, DType::Float32, Device::cuda()};
    const auto updates =
        make_storage<float>(harness, updates_spec, detail::Layout{{1, 2}, 1}, {99, 1, 4, 2, 5, 3, 6, 88});
    const auto indices =
        make_storage<std::int64_t>(harness, indices_spec, detail::Layout{{0, -1}, 3}, {99, 1, 0, 1, 88});
    const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1),
                                                 std::vector<float>(10, -42));
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{updates_spec, updates},
                                                   detail::TensorView{indices_spec, indices}};
    const detail::GatherScatterAddPrimitive primitive{output_spec.shape, 1};
    harness.invoke(primitive, inputs, {output_spec, destination});
    const std::vector<float> expected{-42, 2, 4, 0, 0, 5, 10, 0, 0, -42};
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), expected);
    // A second invocation must clear the previous accumulation.
    harness.invoke(primitive, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), expected);
}

TEST_F(CudaCopyKernelTest, GatherAndScatterAddHandleMultipleBlocksAndTails)
{
    CudaKernelHarness harness;
    const detail::TensorSpec updates_spec{Shape{1003}, DType::Float32, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{1003}, DType::Int64, Device::cuda()};
    const detail::TensorSpec output_spec{Shape{3}, DType::Float32, Device::cuda()};
    std::vector<std::int64_t> index_values(1003);
    for (std::size_t i = 0; i < index_values.size(); ++i)
        index_values[i] = static_cast<std::int64_t>(i % 2);
    const auto updates = make_storage<float>(harness, updates_spec, detail::Layout::contiguous(updates_spec.shape),
                                             std::vector<float>(1003, 1));
    const auto indices =
        make_storage<std::int64_t>(harness, indices_spec, detail::Layout::contiguous(indices_spec.shape), index_values);
    const auto destination = make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape, 1),
                                                 {-42, -42, -42, -42, -42});
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{updates_spec, updates},
                                                   detail::TensorView{indices_spec, indices}};
    harness.invoke(detail::GatherScatterAddPrimitive{output_spec.shape, 0}, inputs, {output_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 502, 501, 0, -42}));

    const auto source =
        make_storage<float>(harness, output_spec, detail::Layout::contiguous(output_spec.shape), {7, 9, 11});
    const auto gathered = make_storage<float>(harness, updates_spec, detail::Layout::contiguous(updates_spec.shape, 1),
                                              std::vector<float>(1005, -42));
    const std::array<detail::TensorView, 2> gather_inputs{detail::TensorView{output_spec, source},
                                                          detail::TensorView{indices_spec, indices}};
    harness.invoke(detail::GatherPrimitive{0, 1}, gather_inputs, {updates_spec, gathered});
    std::vector<float> expected(1005, -42);
    for (std::size_t i = 0; i < index_values.size(); ++i)
    {
        expected[i + 1] = i % 2 == 0 ? 7.0F : 9.0F;
    }
    EXPECT_EQ(harness.copy_to_host<float>(gathered.buffer_ref()), expected);
}

TEST_F(CudaCopyKernelTest, GatherAndScatterAddHandleEmptyIndices)
{
    CudaKernelHarness harness;
    const detail::TensorSpec source_spec{Shape{3}, DType::Float32, Device::cuda()};
    const detail::TensorSpec empty_spec{Shape{0}, DType::Float32, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{0}, DType::Int64, Device::cuda()};
    const auto source =
        make_storage<float>(harness, source_spec, detail::Layout::contiguous(source_spec.shape), {1, 2, 3});
    const auto empty =
        make_storage<float>(harness, empty_spec, detail::Layout::contiguous(empty_spec.shape, 1), {-42, -42});
    const auto indices =
        make_storage<std::int64_t>(harness, indices_spec, detail::Layout::contiguous(indices_spec.shape), {});
    const auto destination = make_storage<float>(harness, source_spec, detail::Layout::contiguous(source_spec.shape, 1),
                                                 std::vector<float>(5, -42));
    const std::array<detail::TensorView, 2> gather_inputs{detail::TensorView{source_spec, source},
                                                          detail::TensorView{indices_spec, indices}};
    harness.invoke(detail::GatherPrimitive{0, 1}, gather_inputs, {empty_spec, empty});
    EXPECT_EQ(harness.copy_to_host<float>(empty.buffer_ref()), (std::vector<float>{-42, -42}));
    const std::array<detail::TensorView, 2> scatter_inputs{detail::TensorView{empty_spec, empty},
                                                           detail::TensorView{indices_spec, indices}};
    harness.invoke(detail::GatherScatterAddPrimitive{source_spec.shape, 0}, scatter_inputs, {source_spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 0, 0, 0, -42}));
    harness.invoke(detail::GatherScatterAddPrimitive{empty_spec.shape, 0}, scatter_inputs, {empty_spec, empty});
    EXPECT_EQ(harness.copy_to_host<float>(empty.buffer_ref()), (std::vector<float>{-42, -42}));
}

TEST_F(CudaCopyKernelTest, GatherAndScatterAddRejectInvalidIndicesWithoutPoisoningTheDevice)
{
    CudaKernelHarness harness;
    const detail::TensorSpec spec{Shape{3}, DType::Float32, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{3}, DType::Int64, Device::cuda()};
    const auto source = make_storage<float>(harness, spec, detail::Layout::contiguous(spec.shape), {1, 2, 3});
    const auto indices =
        make_storage<std::int64_t>(harness, indices_spec, detail::Layout::contiguous(indices_spec.shape), {0, 0, 1});
    const auto destination =
        make_storage<float>(harness, spec, detail::Layout::contiguous(spec.shape, 1), std::vector<float>(5, -42));
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{spec, source},
                                                   detail::TensorView{indices_spec, indices}};
    const detail::GatherPrimitive gather{0, 1};
    const detail::GatherScatterAddPrimitive scatter{spec.shape, 0};
    for (const std::int64_t invalid : {std::int64_t{-1}, std::int64_t{3}, std::numeric_limits<std::int64_t>::max()})
    {
        SCOPED_TRACE(invalid);
        const std::array<std::int64_t, 3> bad{0, invalid, 1};
        harness.copy_from_host<std::int64_t>(indices.buffer_ref(), bad);
        EXPECT_THROW(harness.invoke(gather, inputs, {spec, destination}), std::out_of_range);
        EXPECT_THROW(harness.invoke(scatter, inputs, {spec, destination}), std::out_of_range);
        const auto result = harness.copy_to_host<float>(destination.buffer_ref());
        EXPECT_EQ(result.front(), -42);
        EXPECT_EQ(result.back(), -42);
    }
    const std::array<std::int64_t, 3> valid{0, 1, 2};
    harness.copy_from_host<std::int64_t>(indices.buffer_ref(), valid);
    harness.invoke(gather, inputs, {spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 1, 2, 3, -42}));
    harness.invoke(scatter, inputs, {spec, destination});
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, 1, 2, 3, -42}));
}

TEST_F(CudaCopyKernelTest, RejectsNonemptyIndicesIntoAnEmptyGatherAxis)
{
    CudaKernelHarness harness;
    const detail::TensorSpec empty_spec{Shape{0}, DType::Float32, Device::cuda()};
    const detail::TensorSpec value_spec{Shape{1}, DType::Float32, Device::cuda()};
    const detail::TensorSpec indices_spec{Shape{1}, DType::Int64, Device::cuda()};
    const auto empty = make_storage<float>(harness, empty_spec, detail::Layout::contiguous(empty_spec.shape), {});
    const auto value = make_storage<float>(harness, value_spec, detail::Layout::contiguous(value_spec.shape), {1});
    const auto indices =
        make_storage<std::int64_t>(harness, indices_spec, detail::Layout::contiguous(indices_spec.shape), {0});
    const auto destination =
        make_storage<float>(harness, value_spec, detail::Layout::contiguous(value_spec.shape, 1), {-42, -42, -42});
    const std::array<detail::TensorView, 2> gather_inputs{detail::TensorView{empty_spec, empty},
                                                          detail::TensorView{indices_spec, indices}};
    EXPECT_THROW(harness.invoke(detail::GatherPrimitive{0, 1}, gather_inputs, {value_spec, destination}),
                 std::out_of_range);
    const std::array<detail::TensorView, 2> scatter_inputs{detail::TensorView{value_spec, value},
                                                           detail::TensorView{indices_spec, indices}};
    EXPECT_THROW(
        harness.invoke(detail::GatherScatterAddPrimitive{empty_spec.shape, 0}, scatter_inputs, {empty_spec, empty}),
        std::out_of_range);
    EXPECT_EQ(harness.copy_to_host<float>(destination.buffer_ref()), (std::vector<float>{-42, -42, -42}));
}
} // namespace minitensor::test
