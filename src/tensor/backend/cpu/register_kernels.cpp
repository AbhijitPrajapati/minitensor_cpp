#include "tensor/backend/cpu/register_kernels.hpp"

#include "tensor/backend/cpu/kernels/registrations.hpp"

namespace minitensor::detail::cpu
{
    void register_kernels(KernelRegistry &registry)
    {
        register_full(registry);
        // register_add(registry);
    }
}