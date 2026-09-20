#include "tensor/backend/cpu/kernels/registrations.hpp"

#include "registrations.hpp"

namespace minitensor::detail::cpu
{
    void register_elementwise_kernels(KernelRegistry &registry)
    {
        register_unary_kernels(registry);
        register_binary_kernels(registry);
    }
}
