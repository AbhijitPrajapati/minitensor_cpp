#pragma once

namespace minitensor::detail
{
    class KernelRegistry;

    namespace cpu
    {
        void register_creation_kernels(KernelRegistry &registry);
        void register_elementwise_kernels(KernelRegistry &registry);
        void register_movement_kernels(KernelRegistry &registry);
        void register_reduction_kernels(KernelRegistry &registry);
    }
}
