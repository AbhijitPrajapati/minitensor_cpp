#include <minitensor/types.hpp>

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
TEST(CpuElementwiseKernelTest, HandlesBroadcastNegativeStridesAndOutputOffsets)
{
    CpuKernelHarness harness;
    const detail::TensorSpec lhs_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec rhs_spec{Shape{1, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::BufferRef lhs_buffer = harness.allocate<float>(6);
    const detail::BufferRef rhs_buffer = harness.allocate<float>(3);
    const detail::BufferRef output_buffer = harness.allocate<float>(7);
    const std::array<float, 6> lhs_physical{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::array<float, 3> rhs_values{10.0F, 20.0F, 30.0F};
    std::copy(lhs_physical.begin(), lhs_physical.end(), harness.data<float>(lhs_buffer));
    std::copy(rhs_values.begin(), rhs_values.end(), harness.data<float>(rhs_buffer));
    std::fill_n(harness.data<float>(output_buffer), 7, -1.0F);

    const detail::Materialization lhs_storage{lhs_buffer, detail::Layout{{-3, 1}, 3}};
    const detail::Materialization rhs_storage{rhs_buffer, detail::Layout::contiguous(rhs_spec.shape)};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{lhs_spec, lhs_storage},
                                                   detail::TensorView{rhs_spec, rhs_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    harness.invoke(detail::AddPrimitive{}, inputs, output);

    const std::array<float, 7> expected{-1.0F, 14.0F, 25.0F, 36.0F, 11.0F, 22.0F, 33.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}
} // namespace minitensor::test
