#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/iteration/reduction.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
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

            const ReductionPlan plan{input.shape(), input.layout(), axes};
            assert(plan.output_size() == output.shape().numel());
            if (plan.output_size() == 0)
            {
                return;
            }

            T *output_data = data<T>(output);
            const auto output_offset = static_cast<std::size_t>(output.layout().offset());
            const T *input_data = data<T>(input);

            plan.for_each_output(
                [&](Shape::size_type output_linear,
                    const ReductionPlan::ReductionRange &range)
                {
                    T accumulator{};
                    range.for_each_run(
                        [&](const ReductionPlan::Run &run)
                        {
                            Layout::offset_type input_offset = run.offset;
                            for (Shape::size_type i = 0; i < run.size; ++i)
                            {
                                assert(input_offset >= 0);
                                accumulator += input_data[static_cast<std::size_t>(input_offset)];
                                input_offset += run.stride;
                            }
                        });
                    output_data[output_offset + output_linear] = accumulator;
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
            KernelKey{typeid(SumPrimitive), DeviceType::Cpu},
            run_sum);
    }
}
