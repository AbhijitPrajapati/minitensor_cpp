#include "register_kernels.hpp"

#include "tensor/backend/cuda/kernels/registrations.hpp"

namespace minitensor::detail::cuda
{
void register_kernels(KernelRegistry &registry)
{
    register_creation_kernels(registry);
}
} // namespace minitensor::detail::cuda