// #include "registrations.hpp"

// #include <span>
// #include <stdexcept>
// #include <cassert>

// #include <minitensor/types.hpp>

// #include "tensor/dispatch/kernel_registry.hpp"
// #include "tensor/dispatch/kernel_key.hpp"
// #include "tensor/ops/add.hpp"
// #include "../cpu_buffer.hpp"

// namespace minitensor::detail::cpu
// {
//     namespace
//     {
//         void run_add(DeviceRuntime &device_runtime, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
//         {
//             (void)device_runtime;

//             assert(inputs.size() == 2);

//             (void)dynamic_cast<const AddPrimitive &>(primitive);

//             const TensorView &lhs = inputs[0];
//             const TensorView &rhs = inputs[1];

//             assert(lhs.dtype() == DType::Float32);
//             assert(rhs.dtype() == DType::Float32);
//             assert(output.dtype() == DType::Float32);
//             assert(output.layout().is_contiguous(output.shape()));

//             const std::size_t numel = output.shape().numel();
//             if (numel == 0)
//             {
//                 return;
//             }

//             const auto &lhs_buffer = dynamic_cast<const CpuBuffer &>(lhs.buffer());
//             const auto &rhs_buffer = dynamic_cast<const CpuBuffer &>(rhs.buffer());
//             auto &output_buffer = dynamic_cast<CpuBuffer &>(output.buffer());

//             const auto *lhs_data = reinterpret_cast<const float *>(lhs_buffer.data());
//             const auto *rhs_data = reinterpret_cast<const float *>(rhs_buffer.data());
//             auto *output_data = reinterpret_cast<float *>(output_buffer.data());

//             const auto output_offset = static_cast<std::size_t>(output.layout().offset());

//             // actual buffer traversal logic will go here
//         }
//     }

//     void register_add(KernelRegistry &registry)
//     {
//         KernelKey key{typeid(AddPrimitive), DeviceType::Cpu, DType::Float32};
//         registry.register_kernel(std::move(key), run_add);
//     }
// }