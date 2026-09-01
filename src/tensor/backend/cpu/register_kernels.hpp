#pragma once

namespace minitensor::detail
{
    class KernelRegistry;
    namespace cpu
    {
        void register_kernels(KernelRegistry &registry);
    }
}