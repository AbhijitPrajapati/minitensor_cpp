#include "environment.hpp"

#include <span>

#include <minitensor/types.hpp>

#include "evaluator.hpp"
#include "tensor/backend/cpu/register_kernels.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/graph/fwd.hpp"

#ifdef ENABLE_CUDA
#include "tensor/backend/cuda/register_kernels.hpp"
#endif

namespace minitensor::detail
{
ExecutionEnvironment::ExecutionEnvironment()
{
    cpu::register_kernels(kernels_);
#ifdef ENABLE_CUDA
    cuda::register_kernels(kernels_);
#endif
}

void ExecutionEnvironment::evaluate(std::span<const ValueRef> roots)
{
    Evaluator evaluator(runtimes_, kernels_);
    evaluator.evaluate(roots);
}

DeviceRuntime &ExecutionEnvironment::runtime_for(const Device &device)
{
    return runtimes_.runtime_for(device);
}

ExecutionEnvironment &environment()
{
    static ExecutionEnvironment environment;
    return environment;
}
} // namespace minitensor::detail
