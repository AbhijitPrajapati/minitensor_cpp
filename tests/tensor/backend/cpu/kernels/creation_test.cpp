#include <minitensor/random.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <span>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/support/kernel_harness.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
TEST(CpuCreationKernelTest, FullWritesOffsetAndEmptyOutputs)
{
    CpuKernelHarness harness;
    const detail::TensorSpec spec{Shape{2, 3}, DType::Float32, Device::cpu()};
    const detail::BufferRef buffer = harness.allocate<float>(7);
    std::fill_n(harness.data<float>(buffer), 7, -10.0F);
    const detail::Materialization storage{buffer, detail::Layout{{3, 1}, 1}};
    const detail::MutableTensorView output{spec, storage};
    const detail::FullPrimitive primitive{spec, 2.5F};
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);

    const std::array<float, 7> expected{-10.0F, 2.5F, 2.5F, 2.5F, 2.5F, 2.5F, 2.5F};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), harness.data<float>(buffer)));

    const detail::TensorSpec empty_spec{Shape{2, 0}, DType::Float32, Device::cpu()};
    const detail::BufferRef empty_buffer = harness.allocate<float>(0);
    const detail::Materialization empty_storage{empty_buffer, detail::Layout::contiguous(empty_spec.shape)};
    const detail::MutableTensorView empty_output{empty_spec, empty_storage};
    const detail::FullPrimitive empty_primitive{empty_spec, 4.0F};
    EXPECT_NO_THROW(harness.invoke(empty_primitive, std::span<const detail::TensorView>{}, empty_output));
}

TEST(CpuCreationKernelTest, UniformIsDeterministicAndHonorsOutputOffset)
{
    CpuKernelHarness harness;
    const detail::TensorSpec spec{Shape{5}, DType::Float32, Device::cpu()};
    const detail::BufferRef buffer = harness.allocate<float>(7);
    std::fill_n(harness.data<float>(buffer), 7, -100.0F);
    const detail::Materialization storage{buffer, detail::Layout{{1}, 1}};
    const detail::MutableTensorView output{spec, storage};
    const detail::UniformPrimitive primitive{spec, -2.0F, 3.0F, RandomKey{101}};

    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    EXPECT_FLOAT_EQ(harness.data<float>(buffer)[0], -100.0F);
    EXPECT_FLOAT_EQ(harness.data<float>(buffer)[6], -100.0F);
    EXPECT_TRUE(std::all_of(harness.data<float>(buffer) + 1, harness.data<float>(buffer) + 6,
                            [](float value) { return value >= -2.0F && value < 3.0F; }));

    std::array<float, 5> first{};
    std::copy_n(harness.data<float>(buffer) + 1, first.size(), first.begin());
    harness.invoke(primitive, std::span<const detail::TensorView>{}, output);
    EXPECT_TRUE(std::equal(first.begin(), first.end(), harness.data<float>(buffer) + 1));
}
} // namespace minitensor::test
