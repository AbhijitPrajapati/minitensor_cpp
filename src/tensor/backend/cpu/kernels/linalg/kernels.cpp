#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>

#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/linalg/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/linalg/matmul.hpp"

namespace minitensor::detail::cpu
{
namespace
{
void run_matmul(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    assert(inputs.size() == 2);
    assert(output.layout().is_contiguous(output.shape()));
    (void)dynamic_cast<const MatmulPrimitive &>(primitive);

    dispatch_floating_dtype(
        output.dtype(), [&]<typename T>(std::type_identity<T>) { matmul_dispatch<T>(inputs[0], inputs[1], output); });
}
} // namespace

void register_linalg_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{MatmulPrimitive::ID, DeviceType::Cpu}, run_matmul);
}
} // namespace minitensor::detail::cpu
