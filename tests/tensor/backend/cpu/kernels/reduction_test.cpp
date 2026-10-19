#include <minitensor/types.hpp>

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/reduction/sum.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
TEST(CpuReductionKernelTest, SumHandlesNegativeStridesAndOutputOffsets)
{
    CpuKernelHarness harness;
    const detail::TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2}, DType::Float32, Device::cpu()};
    const detail::BufferRef input_buffer = harness.allocate<float>(6);
    const detail::BufferRef output_buffer = harness.allocate<float>(4);
    const std::array<float, 6> input_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::copy(input_values.begin(), input_values.end(), harness.data<float>(input_buffer));
    std::fill_n(harness.data<float>(output_buffer), 4, -1.0F);

    const detail::Materialization input_storage{input_buffer, detail::Layout{{-3, 1}, 3}};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 1> inputs{detail::TensorView{input_spec, input_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    const std::array<Axis, 1> axes{1};
    harness.invoke(detail::SumPrimitive{axes, 2, false}, inputs, output);

    const std::array<float, 4> expected{-1.0F, 15.0F, 6.0F, -1.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}
} // namespace minitensor::test
