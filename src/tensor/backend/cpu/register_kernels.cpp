#include "tensor/backend/cpu/register_kernels.hpp"

#include "tensor/backend/cpu/kernels/registrations.hpp"

namespace minitensor::detail::cpu
{
    void register_kernels(KernelRegistry &registry)
    {
        register_full(registry);
        register_add(registry);
        register_multiply(registry);
        register_reshape(registry);
        register_sum(registry);
    }
}