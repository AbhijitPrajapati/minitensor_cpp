#include <minitensor/types.hpp>

#include <array>

#include <gtest/gtest.h>

#include "tensor/backend/cuda/register_kernels.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"

namespace minitensor::test
{
TEST(CudaKernelRegistrationTest, RegistersEveryKernelBackedPrimitive)
{
    detail::KernelRegistry registry;
    detail::cuda::register_kernels(registry);
    const std::array<detail::PrimitiveId, 2> primitive_ids{detail::FullPrimitive::ID, detail::ContiguousPrimitive::ID};

    for (const detail::PrimitiveId primitive_id : primitive_ids)
    {
        EXPECT_TRUE(registry.contains(detail::KernelKey{primitive_id, DeviceType::Cuda})) << primitive_id.value();
        EXPECT_FALSE(registry.contains(detail::KernelKey{primitive_id, DeviceType::Cpu})) << primitive_id.value();
    }
}
} // namespace minitensor::test
