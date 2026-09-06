#pragma once

#include <span>

#include "tensor/graph/fwd.hpp"
#include "runtime_registry.hpp"
#include "tensor/dispatch/kernel_registry.hpp"

namespace minitensor::detail
{
    class ExecutionEnvironment final
    {
    public:
        ExecutionEnvironment();
        void evaluate(std::span<const ValueRef> roots);

    private:
        RuntimeRegistry runtimes_;
        KernelRegistry kernels_;
    };

    [[nodiscard]] ExecutionEnvironment &environment();
}