#include <minitensor/types.hpp>

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
TEST(CpuLinalgKernelTest, MatmulHonorsOutputOffset)
{
    CpuKernelHarness harness;
    const detail::TensorSpec lhs_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::TensorSpec rhs_spec{Shape{3, 2}, DType::Float32, Device::cpu()};
    const detail::TensorSpec output_spec{Shape{2, 2}, DType::Float32, Device::cpu()};
    const detail::BufferRef lhs_buffer = harness.allocate<float>(6);
    const detail::BufferRef rhs_buffer = harness.allocate<float>(6);
    const detail::BufferRef output_buffer = harness.allocate<float>(6);
    const std::array<float, 6> lhs_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::array<float, 6> rhs_values{7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    std::copy(lhs_values.begin(), lhs_values.end(), harness.data<float>(lhs_buffer));
    std::copy(rhs_values.begin(), rhs_values.end(), harness.data<float>(rhs_buffer));
    std::fill_n(harness.data<float>(output_buffer), 6, -1.0F);

    const detail::Materialization lhs_storage{lhs_buffer, detail::Layout::contiguous(lhs_spec.shape)};
    const detail::Materialization rhs_storage{rhs_buffer, detail::Layout::contiguous(rhs_spec.shape)};
    const detail::Materialization output_storage{output_buffer, detail::Layout::contiguous(output_spec.shape, 1)};
    const std::array<detail::TensorView, 2> inputs{detail::TensorView{lhs_spec, lhs_storage},
                                                   detail::TensorView{rhs_spec, rhs_storage}};
    const detail::MutableTensorView output{output_spec, output_storage};
    harness.invoke(detail::MatmulPrimitive{}, inputs, output);

    const std::array<float, 6> expected{-1.0F, 58.0F, 64.0F, 139.0F, 154.0F, -1.0F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(output_buffer)));
}
} // namespace minitensor::test
