#include <minitensor/nn/transfer.hpp>

#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>

#include "nn/support/parameter_trees.hpp"

#ifdef MINITENSOR_TEST_ENABLE_CUDA
#include "tensor/backend/cuda/support/device_test.hpp"
#endif

namespace minitensor::nn::test
{
TEST(ParameterTransferTest, PreservesStateWhenConsumingSnapshotOnTheSameDevice)
{
    ParameterBuilder builder;
    const auto first = builder.add({full(Shape{2}, 3.0F)});
    const auto second = builder.add({full(Shape{}, -2.0F), {.trainable = false}});
    auto source = std::move(builder).finish(support::ParameterPair{first, second});

    const auto transferred = to_device(std::move(source), Device::cpu());

    EXPECT_EQ(transferred.tree().first(), first);
    EXPECT_EQ(transferred.tree().second(), second);
    EXPECT_EQ(transferred.parameters().size(), 2U);
    EXPECT_EQ(transferred.value(first).device(), Device::cpu());
    EXPECT_EQ(transferred.value(first).shape(), Shape{2});
    EXPECT_EQ(to_vector(transferred.value(first)), (std::vector<float>{3.0F, 3.0F}));
    EXPECT_EQ(transferred.value(second).shape(), Shape{});
    EXPECT_FLOAT_EQ(item(transferred.value(second)), -2.0F);
    EXPECT_TRUE(transferred.get(first).metadata.trainable);
    EXPECT_FALSE(transferred.get(second).metadata.trainable);
}

TEST(ParameterTransferTest, AcceptsEmptySnapshots)
{
    ParameterBuilder builder;
    const auto source = std::move(builder).finish(support::EmptyTree{});

    const auto transferred = to_device(source, Device::cpu());

    EXPECT_EQ(transferred.parameters().size(), 0U);
    EXPECT_EQ(source.parameters().size(), 0U);
}

TEST(ParameterTransferTest, UnavailableDestinationLeavesTheSourceUnchanged)
{
    ParameterBuilder builder;
    const auto id = builder.add({full(Shape{2}, 4.0F), {.trainable = false}});
    const auto source = std::move(builder).finish(support::ParameterPair{id, id});

    EXPECT_THROW((void)to_device(source, Device::cpu(1)), std::runtime_error);

    EXPECT_EQ(source.parameters().size(), 1U);
    EXPECT_EQ(source.value(id).device(), Device::cpu());
    EXPECT_FALSE(source.get(id).metadata.trainable);
    EXPECT_EQ(to_vector(source.value(id)), (std::vector<float>{4.0F, 4.0F}));
}

#ifdef MINITENSOR_TEST_ENABLE_CUDA
using ParameterCudaTransferTest = ::minitensor::test::CudaDeviceTest;

TEST_F(ParameterCudaTransferTest, RoundTripsValuesPreservingTiesMetadataAndSource)
{
    ParameterBuilder builder;
    const auto shared = builder.add({full(Shape{2}, 3.0F)});
    const auto frozen = builder.add({full(Shape{}, -2, TensorOptions{DType::Int64}), {.trainable = false}});
    const auto source = std::move(builder).finish(support::MixedParameterTree{shared, frozen});

    auto uploaded = to_device(source, Device::cuda());

    EXPECT_EQ(uploaded.tree().first(), shared);
    EXPECT_EQ(uploaded.tree().second(), shared);
    EXPECT_EQ(uploaded.tree().frozen(), frozen);
    EXPECT_EQ(uploaded.parameters().size(), 2U);
    EXPECT_EQ(&uploaded.value(uploaded.tree().first()), &uploaded.value(uploaded.tree().second()));
    EXPECT_EQ(uploaded.value(shared).device(), Device::cuda());
    EXPECT_EQ(uploaded.value(frozen).device(), Device::cuda());
    EXPECT_EQ(uploaded.value(shared).shape(), Shape{2});
    EXPECT_EQ(uploaded.value(frozen).shape(), Shape{});
    EXPECT_EQ(uploaded.value(frozen).dtype(), DType::Int64);
    EXPECT_EQ(to_vector(uploaded.value(shared)), (std::vector<float>{3.0F, 3.0F}));
    EXPECT_EQ(item<std::int64_t>(uploaded.value(frozen)), -2);
    EXPECT_TRUE(uploaded.get(shared).metadata.trainable);
    EXPECT_FALSE(uploaded.get(frozen).metadata.trainable);

    const auto restored = to_device(std::move(uploaded), Device::cpu());
    EXPECT_EQ(restored.tree().first(), shared);
    EXPECT_EQ(restored.tree().second(), shared);
    EXPECT_EQ(restored.tree().frozen(), frozen);
    EXPECT_EQ(restored.parameters().size(), 2U);
    for (const auto &[id, state] : restored.parameters())
    {
        EXPECT_EQ(state.value.device(), Device::cpu());
        EXPECT_EQ(state.value.shape(), source.value(id).shape());
        EXPECT_EQ(state.value.dtype(), source.value(id).dtype());
        EXPECT_EQ(state.metadata, source.get(id).metadata);
        EXPECT_EQ(source.value(id).device(), Device::cpu());
    }
    EXPECT_EQ(to_vector(restored.value(shared)), to_vector(source.value(shared)));
    EXPECT_EQ(item<std::int64_t>(restored.value(frozen)), item<std::int64_t>(source.value(frozen)));
}
#endif
} // namespace minitensor::nn::test
