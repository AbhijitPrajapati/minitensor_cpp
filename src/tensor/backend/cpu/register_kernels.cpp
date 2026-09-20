#include "tensor/backend/cpu/register_kernels.hpp"

#include "tensor/backend/cpu/kernels/registrations.hpp"

namespace minitensor::detail::cpu
{
    void register_kernels(KernelRegistry &registry)
    {
        register_creation_kernels(registry);
        register_elementwise_kernels(registry);
        register_movement_kernels(registry);
        register_reduction_kernels(registry);
    }
}
