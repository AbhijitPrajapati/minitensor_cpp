#pragma once

namespace minitensor::detail
{
    class KernelRegistry;

    namespace cpu
    {
        void register_unary_kernels(KernelRegistry &registry);
        void register_binary_kernels(KernelRegistry &registry);
    }
}
