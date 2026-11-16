#include <minitensor/types.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/execution/runtime_pool.hpp"
#include "tensor/support/test_runtime.hpp"

#ifdef MINITENSOR_TEST_ENABLE_CUDA
#include "tensor/backend/cuda/cuda_runtime.hpp"
#endif

namespace minitensor::test
{
TEST(RuntimePoolTest, ProvidesCpuZeroWithoutExplicitRegistration)
{
    const detail::RuntimePool pool;
    detail::DeviceRuntime &runtime = pool.runtime_for(Device::cpu());
    EXPECT_NE(dynamic_cast<detail::cpu::CpuRuntime *>(&runtime), nullptr);
    EXPECT_EQ(runtime.device(), Device::cpu());
    EXPECT_EQ(&pool.runtime_for(Device::cpu()), &runtime);
}

TEST(RuntimePoolTest, OwnsAndRetrievesRuntimesByDevice)
{
    detail::RuntimePool pool;
    auto runtime = std::make_unique<TestRuntime>(Device::cpu(3));
    TestRuntime *address = runtime.get();
    pool.register_runtime(std::move(runtime));

    EXPECT_EQ(runtime, nullptr);
    EXPECT_EQ(pool.runtime_for(Device::cpu(3)).device(), Device::cpu(3));
    EXPECT_EQ(&pool.runtime_for(Device::cpu(3)), address);
    const detail::RuntimePool &const_pool = pool;
    EXPECT_EQ(&const_pool.runtime_for(Device::cpu(3)), address);
}

TEST(RuntimePoolTest, DistinguishesDeviceIndices)
{
    detail::RuntimePool pool;
    auto first = std::make_unique<TestRuntime>(Device::cpu(2));
    auto second = std::make_unique<TestRuntime>(Device::cpu(1));
    TestRuntime *first_address = first.get();
    TestRuntime *second_address = second.get();
    pool.register_runtime(std::move(first));
    pool.register_runtime(std::move(second));

    EXPECT_EQ(&pool.runtime_for(Device::cpu(2)), first_address);
    EXPECT_EQ(&pool.runtime_for(Device::cpu(1)), second_address);
}

TEST(RuntimePoolTest, RejectsMissingEmptyAndDuplicateEntries)
{
    detail::RuntimePool pool;
    EXPECT_THROW((void)pool.runtime_for(Device::cpu(1)), std::runtime_error);
    EXPECT_THROW(pool.register_runtime(std::unique_ptr<detail::DeviceRuntime>{}), std::invalid_argument);

    EXPECT_THROW(pool.register_runtime(std::make_unique<TestRuntime>()), std::logic_error);
}

TEST(RuntimePoolTest, DestroysOwnedRuntimes)
{
    bool destroyed = false;
    {
        detail::RuntimePool pool;
        pool.register_runtime(std::make_unique<TestRuntime>(Device::cpu(3), &destroyed));
        EXPECT_FALSE(destroyed);
    }
    EXPECT_TRUE(destroyed);
}

#ifdef MINITENSOR_TEST_ENABLE_CUDA
TEST(RuntimePoolTest, LazilyCreatesAndCachesCudaRuntimesByDevice)
{
    // Runtime construction does not allocate GPU memory or require a driver.
    const detail::RuntimePool pool;
    detail::DeviceRuntime &first = pool.runtime_for(Device::cuda());
    detail::DeviceRuntime &second = pool.runtime_for(Device::cuda(1));
    EXPECT_NE(dynamic_cast<detail::cuda::CudaRuntime *>(&first), nullptr);
    EXPECT_NE(dynamic_cast<detail::cuda::CudaRuntime *>(&second), nullptr);
    EXPECT_EQ(first.device(), Device::cuda());
    EXPECT_EQ(second.device(), Device::cuda(1));
    EXPECT_NE(&first, &second);
    EXPECT_EQ(&pool.runtime_for(Device::cuda()), &first);
    EXPECT_EQ(&pool.runtime_for(Device::cuda(1)), &second);
}

TEST(RuntimePoolTest, KeepsRuntimeReferencesValidWhenTheCacheGrows)
{
    const detail::RuntimePool pool;
    detail::DeviceRuntime *cpu = &pool.runtime_for(Device::cpu());
    detail::DeviceRuntime *cuda = &pool.runtime_for(Device::cuda());
    for (std::uint32_t index = 1; index < 64; ++index)
    {
        EXPECT_EQ(pool.runtime_for(Device::cuda(index)).device(), Device::cuda(index));
    }
    EXPECT_EQ(&pool.runtime_for(Device::cpu()), cpu);
    EXPECT_EQ(&pool.runtime_for(Device::cuda()), cuda);
}
#else
TEST(RuntimePoolTest, RejectsCudaWhenTheBackendIsDisabled)
{
    const detail::RuntimePool pool;
    EXPECT_THROW((void)pool.runtime_for(Device::cuda()), std::runtime_error);
    EXPECT_THROW((void)pool.runtime_for(Device::cuda(1)), std::runtime_error);
    EXPECT_EQ(pool.runtime_for(Device::cpu()).device(), Device::cpu());
}
#endif
} // namespace minitensor::test
