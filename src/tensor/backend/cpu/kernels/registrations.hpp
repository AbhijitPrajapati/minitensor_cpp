#pragma once

namespace minitensor::detail
{
    class KernelRegistry;

    namespace cpu
    {
        void register_creation_kernels(KernelRegistry &registry);
        void register_elementwise_kernels(KernelRegistry &registry);
        void register_copy_kernels(KernelRegistry &registry);
        void register_reduction_kernels(KernelRegistry &registry);
        void register_linalg_kernels(KernelRegistry &registry);
    }
}
