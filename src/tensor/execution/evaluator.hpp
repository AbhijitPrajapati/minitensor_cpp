#pragma once

#include <span>

#include "tensor/graph/fwd.hpp"

namespace minitensor::detail
{
    class RuntimeRegistry;
    class KernelRegistry;

    class Evaluator final
    {
    public:
        Evaluator(RuntimeRegistry &runtimes, const KernelRegistry &kernels) noexcept;
        void evaluate(std::span<const ValueRef> roots);

    private:
        const RuntimeRegistry &runtimes_;
        const KernelRegistry &kernels_;
    };
}