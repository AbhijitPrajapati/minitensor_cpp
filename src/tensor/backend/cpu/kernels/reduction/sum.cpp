#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/buffer_access.hpp"
#include "tensor/backend/cpu/kernels/common/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/iteration/reduction.hpp"
#include "tensor/primitives/reduction/sum.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        template <CpuElement T>
        void sum_values(
            const TensorView &input,
            MutableTensorView output,
            std::span<const Shape::size_type> axes)
        {
            assert(input.dtype() == ElementDType<T>::value);
            assert(output.dtype() == ElementDType<T>::value);
            assert(output.layout().is_contiguous(output.shape()));

            if (output.shape().numel() == 0)
            {
                return;
            }

            const T *input_data = data<T>(input);
            T *output_data = data<T>(output);
            const auto output_offset = static_cast<std::size_t>(output.layout().offset());

            const ReductionPlan plan(input.shape(), input.layout(), output.shape(), axes);
            std::fill_n(output_data + output_offset, plan.output_numel(), T{0});

            plan.for_each(
                [&](Shape::size_type output_linear, Layout::offset_type input_offset)
                {
                    assert(input_offset >= 0);
                    output_data[output_offset + output_linear] +=
                        input_data[static_cast<std::size_t>(input_offset)];
                });
        }

        void run_sum(
            DeviceRuntime &,
            const Primitive &primitive,
            std::span<const TensorView> inputs,
            MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &sum = dynamic_cast<const SumPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    sum_values<T>(inputs.front(), output, sum.axes());
                });
        }
    }

    void register_reduction_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(
            KernelKey{typeid(SumPrimitive), DeviceType::Cpu, DType::Float32},
            run_sum);
    }
}
