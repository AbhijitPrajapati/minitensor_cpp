#pragma once

namespace minitensor::detail
{
class KernelRegistry;
namespace cuda
{
void register_kernels(KernelRegistry &registry);
}
} // namespace minitensor::detail