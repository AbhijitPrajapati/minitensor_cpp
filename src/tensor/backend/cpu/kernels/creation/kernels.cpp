#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/creation/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_full(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.empty());
            (void)inputs;

            const auto &full = dynamic_cast<const FullPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    fill(output, static_cast<T>(full.fill_value()));
                });
        }

        void run_uniform(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.empty());
            (void)inputs;

            const auto &uniform_primitive = dynamic_cast<const UniformPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    const auto low = static_cast<T>(uniform_primitive.low());
                    const auto high = static_cast<T>(uniform_primitive.high());
                    uniform(output, low, high, uniform_primitive.key());
                });
        }

        void run_normal(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.empty());
            (void)inputs;

            const auto &normal_primitive = dynamic_cast<const NormalPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    const auto mean = static_cast<T>(normal_primitive.mean());
                    const auto std_dev = static_cast<T>(normal_primitive.std_dev());
                    normal(output, mean, std_dev, normal_primitive.key());
                });
        }
    }

    void register_creation_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(FullPrimitive), DeviceType::Cpu}, run_full);
        registry.register_kernel(KernelKey{typeid(UniformPrimitive), DeviceType::Cpu}, run_uniform);
        registry.register_kernel(KernelKey{typeid(NormalPrimitive), DeviceType::Cpu}, run_normal);
    }
}
