#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/buffer_access.hpp"
#include "tensor/backend/cpu/kernels/common/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_full(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.empty());
            assert(output.layout().is_contiguous(output.shape()));

            const auto &full = dynamic_cast<const FullPrimitive &>(primitive);
            const std::size_t numel = output.shape().numel();
            if (numel == 0)
            {
                return;
            }

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    T *output_data = data<T>(output);
                    const auto output_offset = static_cast<std::size_t>(output.layout().offset());
                    std::fill_n(output_data + output_offset, numel, static_cast<T>(full.fill_value()));
                });
        }
    }

    void register_creation_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(FullPrimitive), DeviceType::Cpu}, run_full);
    }
}
