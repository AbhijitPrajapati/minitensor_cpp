#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#ifdef MINITENSOR_TEST_ENABLE_CUDA
#include "tensor/backend/cuda/support/device_test.hpp"
#endif

namespace minitensor::test
{
TEST(DataTest, RoundTripsHostDataByValue)
{
    std::array<float, 6> source{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const std::array<float, 6> expected = source;
    const Tensor tensor = from_data(source, Shape{2, 3});

    EXPECT_EQ(tensor.shape(), (Shape{2, 3}));
    EXPECT_EQ(tensor.dtype(), DType::Float32);
    EXPECT_EQ(tensor.device(), Device::cpu());

    source[0] = -100.0F;
    EXPECT_EQ(to_vector(tensor), std::vector<float>(expected.begin(), expected.end()));
}

TEST(DataTest, ReadsViewsInLogicalOrder)
{
    const Tensor matrix = from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3});
    const Tensor transposed = transpose(matrix);
    const std::vector<float> expected{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
    EXPECT_EQ(to_vector(transposed), expected);
}

TEST(DataTest, RoundTripsInt64DataWithoutFloatingPointConversion)
{
    const std::array<std::int64_t, 3> source{9'007'199'254'740'993, -9'007'199'254'740'993, 42};
    const Tensor tensor = from_data(source, Shape{3});

    EXPECT_EQ(tensor.dtype(), DType::Int64);
    EXPECT_EQ(to_vector<std::int64_t>(tensor), (std::vector<std::int64_t>{source.begin(), source.end()}));
    EXPECT_EQ(item<std::int64_t>(from_data<std::int64_t>({17}, Shape{})), 17);
    EXPECT_THROW((void)to_vector<float>(tensor), std::invalid_argument);
}

TEST(DataTest, SupportsScalarsSingletonsAndEmptyTensors)
{
    EXPECT_FLOAT_EQ(item(from_data({-3.25F}, Shape{})), -3.25F);
    EXPECT_FLOAT_EQ(item(full(Shape{1, 1}, 7.5F)), 7.5F);

    const Tensor empty = from_data(std::span<const float>{}, Shape{2, 0, 3});
    EXPECT_TRUE(to_vector(empty).empty());
}

TEST(DataTest, ValidatesElementCounts)
{
    const std::array<float, 2> values{1.0F, 2.0F};
    EXPECT_THROW((void)from_data(values, Shape{3}), std::invalid_argument);
    EXPECT_THROW((void)item(full(Shape{2}, 1.0F)), std::invalid_argument);
}

TEST(DataTest, ReportsUnavailableDevices)
{
    const std::array<float, 1> values{1.0F};
    EXPECT_THROW((void)from_data(values, Shape{1}, Device::cpu(1)), std::runtime_error);
}

TEST(DataTest, SameDeviceTransferPreservesValuesAndDoesNotRequireEvaluation)
{
    const Tensor view = transpose(from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3}));
    const Tensor transferred = to_device(view, Device::cpu());
    EXPECT_EQ(transferred.shape(), (Shape{3, 2}));
    EXPECT_EQ(transferred.device(), Device::cpu());
    EXPECT_EQ(to_vector(transferred), (std::vector<float>{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));

    // An unavailable runtime would fail if the same-device path tried to evaluate.
    const Tensor lazy = full(Shape{2}, 3.0F, TensorOptions{DType::Float32, Device::cpu(1)});
    EXPECT_NO_THROW((void)to_device(lazy, Device::cpu(1)));
}

TEST(DataTest, TransferReportsUnavailableDestinations)
{
    const Tensor source = from_data({1.0F, 2.0F}, Shape{2});
    EXPECT_THROW((void)to_device(source, Device::cpu(1)), std::runtime_error);
    EXPECT_EQ(to_vector(source), (std::vector<float>{1.0F, 2.0F}));

    const Tensor empty = from_data(std::span<const float>{}, Shape{0});
    EXPECT_THROW((void)to_device(empty, Device::cpu(1)), std::runtime_error);
}

#ifdef MINITENSOR_TEST_ENABLE_CUDA
using DataCudaTest = CudaDeviceTest;

TEST_F(DataCudaTest, TransfersHostDataAndLazyTensorsToCuda)
{
    std::array<float, 3> values{1.0F, -2.0F, 3.5F};
    const Tensor uploaded = from_data(values, Shape{3}, Device::cuda());
    values[0] = 100.0F;
    EXPECT_EQ(uploaded.device(), Device::cuda());
    EXPECT_EQ(to_vector(uploaded), (std::vector<float>{1.0F, -2.0F, 3.5F}));

    const Tensor source = full(Shape{2, 3}, 7.5F);
    const Tensor transferred = to_device(source, Device::cuda());
    EXPECT_EQ(transferred.shape(), source.shape());
    EXPECT_EQ(transferred.dtype(), source.dtype());
    EXPECT_EQ(transferred.device(), Device::cuda());
    EXPECT_EQ(to_vector(transferred), std::vector<float>(6, 7.5F));

    const Tensor restored = to_device(transferred, Device::cpu());
    EXPECT_EQ(restored.shape(), source.shape());
    EXPECT_EQ(restored.dtype(), source.dtype());
    EXPECT_EQ(restored.device(), Device::cpu());
    EXPECT_EQ(to_vector(restored), std::vector<float>(6, 7.5F));
}

TEST_F(DataCudaTest, TransfersViewsInLogicalOrderInBothDirections)
{
    const Tensor source = from_data({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}, Shape{3, 3});
    const Tensor host_view = transpose(slice(source, 0, 1, 3));
    const Tensor uploaded = to_device(host_view, Device::cuda());
    EXPECT_EQ(uploaded.shape(), (Shape{3, 2}));
    EXPECT_EQ(to_vector(uploaded), (std::vector<float>{3.0F, 6.0F, 4.0F, 7.0F, 5.0F, 8.0F}));

    const Tensor device_source = to_device(source, Device::cuda());
    const Tensor device_view = transpose(slice(device_source, 0, 1, 3));
    EXPECT_EQ(to_vector(device_view), to_vector(host_view));
    const Tensor restored = to_device(device_view, Device::cpu());
    EXPECT_EQ(restored.shape(), host_view.shape());
    EXPECT_EQ(to_vector(restored), to_vector(host_view));

    // A contiguous view can still start at a nonzero buffer offset.
    EXPECT_EQ(to_vector(to_device(select(source, 0, 1), Device::cuda())), (std::vector<float>{3.0F, 4.0F, 5.0F}));
    EXPECT_EQ(to_vector(to_device(select(device_source, 0, 1), Device::cpu())), (std::vector<float>{3.0F, 4.0F, 5.0F}));
}

TEST_F(DataCudaTest, PreservesInt64ValuesAcrossDevices)
{
    const std::array<std::int64_t, 3> values{9'007'199'254'740'993, -9'007'199'254'740'993, 42};
    const Tensor source = from_data(values, Shape{3});
    const Tensor uploaded = to_device(source, Device::cuda());
    const Tensor restored = to_device(uploaded, Device::cpu());

    EXPECT_EQ(uploaded.dtype(), DType::Int64);
    EXPECT_EQ(restored.dtype(), DType::Int64);
    EXPECT_EQ(to_vector<std::int64_t>(uploaded), (std::vector<std::int64_t>{values.begin(), values.end()}));
    EXPECT_EQ(to_vector<std::int64_t>(restored), (std::vector<std::int64_t>{values.begin(), values.end()}));
}

TEST_F(DataCudaTest, TransfersLazyCudaScalarsAndEmptyTensors)
{
    const Tensor scalar = full(Shape{}, -3.25F, TensorOptions{DType::Float32, Device::cuda()});
    const Tensor restored = to_device(scalar, Device::cpu());
    EXPECT_EQ(restored.shape(), Shape{});
    EXPECT_FLOAT_EQ(item(restored), -3.25F);
    EXPECT_FLOAT_EQ(item(to_device(restored, Device::cuda())), -3.25F);

    const Tensor empty = from_data(std::span<const float>{}, Shape{2, 0, 3});
    const Tensor uploaded = to_device(empty, Device::cuda());
    EXPECT_EQ(uploaded.shape(), empty.shape());
    EXPECT_EQ(uploaded.device(), Device::cuda());
    EXPECT_TRUE(to_vector(uploaded).empty());
    const Tensor empty_restored = to_device(uploaded, Device::cpu());
    EXPECT_EQ(empty_restored.shape(), empty.shape());
    EXPECT_EQ(empty_restored.device(), Device::cpu());
    EXPECT_TRUE(to_vector(empty_restored).empty());
}

TEST_F(DataCudaTest, TransfersBetweenCudaDevicesAndRestoresCurrentDevice)
{
    if (device_count_ < 2)
    {
        GTEST_SKIP() << "Two CUDA devices are required";
    }

    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    const Tensor source = from_data({1.0F, -2.0F, 3.5F}, Shape{3});
    const Tensor first = to_device(source, Device::cuda(0));
    const Tensor second = to_device(first, Device::cuda(1));
    EXPECT_EQ(second.device(), Device::cuda(1));
    EXPECT_EQ(to_vector(second), to_vector(source));
    EXPECT_EQ(to_vector(to_device(second, Device::cuda(0))), to_vector(source));

    int current_device = -1;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    EXPECT_EQ(current_device, 1);
}
#endif
} // namespace minitensor::test
