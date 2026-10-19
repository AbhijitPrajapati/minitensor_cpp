#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/gather.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
TEST(CpuCopyKernelTest, CopiesNegativeStridedInputsInLogicalOrder)
{
    CpuKernelHarness harness;
    const detail::TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{3, 2}, DType::Float32, Device::cpu()};
    const detail::BufferRef input_buffer = harness.allocate<float>(6);
    const detail::BufferRef output_buffer = harness.allocate<float>(7);
    const std::array<float, 6> physical{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::copy(physical.begin(), physical.end(), harness.data<float>(input_buffer));
    std::fill_n(harness.data<float>(output_buffer), 7, -1.0F);

    const detail::Materialization input_storage{input_buffer, detail::Layout{{-3, 1}, 3}};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, input_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    harness.invoke(detail::ReshapePrimitive{output_spec.shape}, inputs, output);

    const std::array<float, 7> expected{-1.0F, 4.0F, 5.0F, 6.0F, 1.0F, 2.0F, 3.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}

TEST(CpuCopyKernelTest, ConcatenatesStridedAndEmptySections)
{
    CpuKernelHarness harness;
    const detail::TensorSpec column_spec{Shape{2, 1}, DType::Float32, Device::cpu()};
    const detail::TensorSpec empty_spec{Shape{2, 0}, DType::Float32, Device::cpu()};
    const detail::TensorSpec block_spec{Shape{2, 2}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::BufferRef column_buffer = harness.allocate<float>(2);
    const detail::BufferRef empty_buffer = harness.allocate<float>(0);
    const detail::BufferRef block_buffer = harness.allocate<float>(4);
    const detail::BufferRef output_buffer = harness.allocate<float>(7);
    const std::array<float, 2> column_values{1.0F, 4.0F};
    const std::array<float, 4> block_physical{2.0F, 5.0F, 3.0F, 6.0F};
    std::copy(column_values.begin(), column_values.end(), harness.data<float>(column_buffer));
    std::copy(block_physical.begin(), block_physical.end(), harness.data<float>(block_buffer));
    std::fill_n(harness.data<float>(output_buffer), 7, -1.0F);

    const detail::Materialization column_storage{column_buffer, detail::Layout::contiguous(column_spec.shape)};
    const detail::Materialization empty_storage{empty_buffer, detail::Layout::contiguous(empty_spec.shape)};
    const detail::Materialization block_storage{block_buffer, detail::Layout{{1, 2}}};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 3> inputs{detail::TensorView{column_spec, column_storage},
                                                   detail::TensorView{empty_spec, empty_storage},
                                                   detail::TensorView{block_spec, block_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    harness.invoke(detail::ConcatenatePrimitive{1, 2}, inputs, output);

    const std::array<float, 7> expected{-1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}

TEST(CpuCopyKernelTest, ScattersSlicesWithNegativeSteps)
{
    CpuKernelHarness harness;
    const detail::TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2, 5}, DType::Float32, Device::cpu()};
    const detail::BufferRef input_buffer = harness.allocate<float>(6);
    const detail::BufferRef output_buffer = harness.allocate<float>(11);
    const std::array<float, 6> input_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::copy(input_values.begin(), input_values.end(), harness.data<float>(input_buffer));
    std::fill_n(harness.data<float>(output_buffer), 11, -1.0F);

    const detail::Materialization input_storage{input_buffer, detail::Layout::contiguous(input_spec.shape)};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, input_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    const detail::BasicIndexParameters parameters =
        detail::BasicIndexParameters::for_slice(1, std::nullopt, std::nullopt, -2, output_spec.shape);
    harness.invoke(detail::BasicIndexScatterPrimitive{parameters}, inputs, output);

    const std::array<float, 11> expected{-1.0F, 3.0F, 0.0F, 2.0F, 0.0F, 1.0F, 6.0F, 0.0F, 5.0F, 0.0F, 4.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}

TEST(CpuCopyKernelTest, ScattersRankReducedSelections)
{
    CpuKernelHarness harness;
    const detail::TensorSpec input_spec{Shape{2, 2}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2, 3, 2}, DType::Float32, Device::cpu()};
    const detail::BufferRef input_buffer = harness.allocate<float>(4);
    const detail::BufferRef output_buffer = harness.allocate<float>(13);
    const std::array<float, 4> input_values{1.0F, 2.0F, 3.0F, 4.0F};
    std::copy(input_values.begin(), input_values.end(), harness.data<float>(input_buffer));
    std::fill_n(harness.data<float>(output_buffer), 13, -1.0F);

    const detail::Materialization input_storage{input_buffer, detail::Layout::contiguous(input_spec.shape)};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, input_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    const detail::BasicIndexParameters parameters = detail::BasicIndexParameters::for_select(1, 1, output_spec.shape);
    harness.invoke(detail::BasicIndexScatterPrimitive{parameters}, inputs, output);

    const std::array<float, 13> expected{-1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}

TEST(CpuCopyKernelTest, GatherHandlesStridedSourcesAndOutputOffsets)
{
    CpuKernelHarness harness;
    const detail::TensorSpec source_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec indices_spec{Shape{2, 2}, DType::Int64, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2, 2}, DType::Float32, Device::cpu()};
    const detail::BufferRef source_buffer = harness.allocate<float>(6);
    const detail::BufferRef indices_buffer = harness.allocate<std::int64_t>(4);
    const detail::BufferRef output_buffer = harness.allocate<float>(6);
    const std::array<float, 6> source_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::array<std::int64_t, 4> index_values{2, 0, 1, 1};
    std::copy(source_values.begin(), source_values.end(), harness.data<float>(source_buffer));
    std::copy(index_values.begin(), index_values.end(), harness.data<std::int64_t>(indices_buffer));
    std::fill_n(harness.data<float>(output_buffer), 6, -1.0F);

    const detail::Materialization source_storage{source_buffer, detail::Layout{{-3, 1}, 3}};
    const detail::Materialization indices_storage{indices_buffer, detail::Layout::contiguous(indices_spec.shape)};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{source_spec, source_storage},
                                                   detail::TensorView{indices_spec, indices_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    harness.invoke(detail::GatherPrimitive{1, 2}, inputs, output);

    const std::array<float, 6> expected{-1.0F, 6.0F, 4.0F, 2.0F, 2.0F, -1.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}
} // namespace minitensor::test
