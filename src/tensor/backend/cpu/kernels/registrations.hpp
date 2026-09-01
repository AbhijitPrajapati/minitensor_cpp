#pragma once

namespace minitensor::detail
{
    class KernelRegistry;

    namespace cpu
    {
        void register_full(KernelRegistry &registry);
        // void register_add(KernelRegistry &registry);
    }
}