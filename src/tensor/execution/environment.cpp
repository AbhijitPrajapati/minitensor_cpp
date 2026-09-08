#include "environment.hpp"

#include <memory>

#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/cpu/register_kernels.hpp"
#include "evaluator.hpp"

namespace minitensor::detail
{
    ExecutionEnvironment::ExecutionEnvironment()
    {
        runtimes_.register_runtime(std::make_unique<cpu::CpuRuntime>());
        cpu::register_kernels(kernels_);
    }

    void ExecutionEnvironment::evaluate(std::span<const ValueRef> roots)
    {
        Evaluator evaluator(runtimes_, kernels_);
        evaluator.evaluate(roots);
    }

    DeviceRuntime &ExecutionEnvironment::runtime_for(const Device &device)
    {
        return runtimes_.get(device);
    }

    ExecutionEnvironment &environment()
    {
        static ExecutionEnvironment environment;
        return environment;
    }
}
