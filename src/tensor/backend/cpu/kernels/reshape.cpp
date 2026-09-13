#include "registrations.hpp"

#include <cassert>
#include <array>
#include <span>
#include <cstddef>
#include <utility>
#include <typeinfo>

#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/ops/reshape.hpp"
#include "tensor/storage/layout.hpp"
#include "../cpu_buffer.hpp"
#include "tensor/iteration/elementwise.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        void run_reshape(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            assert(output.dtype() == DType::Float32);

            assert(dynamic_cast<const ReshapePrimitive *>(&primitive) != nullptr);

            const TensorView &input = inputs.front();
            assert(input.shape().numel() == output.shape().numel());

            assert(output.layout().is_contiguous(output.shape()));

            const auto &input_buffer = dynamic_cast<const CpuBuffer &>(input.buffer());
            auto &output_buffer = dynamic_cast<CpuBuffer &>(output.buffer());

            const auto *input_data = reinterpret_cast<const float *>(input_buffer.data());
            auto *output_data = reinterpret_cast<float *>(output_buffer.data());

            const auto output_offset = static_cast<std::size_t>(output.layout().offset());

            const std::array<Layout, 1> layouts{input.layout()};
            const ElementwisePlan plan(input.shape(), layouts);
            plan.for_each(
                [&](Shape::size_type linear, std::span<const Layout::offset_type> offsets)
                {
                    assert(offsets.size() == 1);
                    assert(offsets.front() >= 0);
                    const auto input_offset = static_cast<std::size_t>(offsets.front());
                    output_data[linear + output_offset] = input_data[input_offset];
                });
        }
    }

    void register_reshape(KernelRegistry &registry)
    {
        KernelKey key{typeid(ReshapePrimitive), DeviceType::Cpu, DType::Float32};
        registry.register_kernel(std::move(key), run_reshape);
    }
}
