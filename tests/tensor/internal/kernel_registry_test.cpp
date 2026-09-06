#include <minitensor/types.hpp>

#include <span>
#include <stdexcept>
#include <typeinfo>
#include <utility>

#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"
#include "../support/test_buffer.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_kernel_registry_test()
    {
        using detail::DeviceRuntime;
        using detail::KernelFn;
        using detail::KernelKey;
        using detail::KernelRegistry;
        using detail::Layout;
        using detail::Materialization;
        using detail::MutableTensorView;
        using detail::Primitive;
        using detail::TensorSpec;
        using detail::TensorView;
        using detail::cpu::CpuRuntime;

        KernelKey key{typeid(IdentitySpecPrimitive), DeviceType::Cpu, DType::Float32};
        const KernelKey equal_key{typeid(IdentitySpecPrimitive), DeviceType::Cpu, DType::Float32};
        KernelKey other_key{typeid(DestructionTrackedPrimitive), DeviceType::Cpu, DType::Float32};
        expect(key == equal_key, "equal kernel keys compare equal");
        expect(key != other_key, "the primitive type participates in kernel key equality");

        KernelRegistry registry;
        expect(!registry.contains(key), "a new registry does not contain a kernel key");
        expect_throws<std::runtime_error>(
            [&registry, &key]
            {
                (void)registry.get(key);
            },
            "getting an unregistered kernel throws");

        expect_throws<std::invalid_argument>(
            [&registry, &other_key]
            {
                registry.register_kernel(other_key, KernelFn{});
            },
            "registering an empty kernel throws");
        expect(!registry.contains(other_key), "rejecting an empty kernel leaves the key unregistered");

        int primary_calls = 0;
        KernelFn primary_kernel = [&primary_calls](
                                      DeviceRuntime &,
                                      const Primitive &,
                                      std::span<const TensorView>,
                                      MutableTensorView)
        {
            ++primary_calls;
        };
        registry.register_kernel(key, std::move(primary_kernel));
        expect(registry.contains(key), "registering a kernel makes its key available");

        const TensorSpec scalar_spec{Shape{}, DType::Float32, Device::cpu()};
        const Materialization scalar_materialization{make_test_buffer(sizeof(float)), Layout{}};
        const MutableTensorView scalar_output{scalar_spec, scalar_materialization};
        CpuRuntime runtime;
        const IdentitySpecPrimitive primitive;

        const KernelFn &retrieved = registry.get(key);
        expect(static_cast<bool>(retrieved), "get returns a callable kernel");
        retrieved(runtime, primitive, std::span<const TensorView>{}, scalar_output);
        expect(primary_calls == 1, "get returns the kernel registered for the requested key");

        int replacement_calls = 0;
        KernelFn replacement = [&replacement_calls](
                                   DeviceRuntime &,
                                   const Primitive &,
                                   std::span<const TensorView>,
                                   MutableTensorView)
        {
            ++replacement_calls;
        };
        expect_throws<std::logic_error>(
            [&registry, &key, &replacement]
            {
                registry.register_kernel(key, replacement);
            },
            "registering a duplicate kernel key throws");

        registry.get(key)(runtime, primitive, std::span<const TensorView>{}, scalar_output);
        expect(primary_calls == 2 && replacement_calls == 0,
               "duplicate registration does not replace the original kernel");
    }
}
