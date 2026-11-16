#pragma once

#include <span>

#include "tensor/graph/fwd.hpp"

namespace minitensor::detail
{
class RuntimePool;
class KernelRegistry;

class Evaluator final
{
  public:
    Evaluator(RuntimePool &runtimes, const KernelRegistry &kernels) noexcept;
    void evaluate(std::span<const ValueRef> roots);

  private:
    const RuntimePool &runtimes_;
    const KernelRegistry &kernels_;
};
} // namespace minitensor::detail