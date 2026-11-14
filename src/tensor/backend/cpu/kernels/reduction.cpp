#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/reduction.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
namespace
{
template <TensorElement T, typename Accumulator, typename Operation>
    requires std::invocable<Operation &, Accumulator, T> &&
             std::convertible_to<std::invoke_result_t<Operation &, Accumulator, T>, Accumulator> &&
             std::convertible_to<Accumulator, T>
void reduce_values(const TensorView &input, MutableTensorView output, std::span<const Shape::size_type> axes,
                   Accumulator initial_value, Operation &&operation)
{
    assert(input.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));

    const ReductionPlan plan{input.shape(), input.layout(), axes};
    assert(plan.output_size() == output.shape().numel());
    if (plan.output_size() == 0)
    {
        return;
    }

    const T *input_data = data<T>(input);
    T *output_data = data<T>(output);
    const auto output_offset = static_cast<std::size_t>(output.layout().offset());

    plan.for_each_output([&](Shape::size_type output_linear, const ReductionPlan::ReductionRange &range) {
        Accumulator accumulator = initial_value;
        range.for_each_run([&](const ReductionPlan::Run &run) {
            Layout::offset_type input_offset = run.offset;
            for (Shape::size_type i = 0; i < run.size; ++i)
            {
                assert(input_offset >= 0);
                accumulator = static_cast<Accumulator>(
                    std::invoke(operation, accumulator, input_data[static_cast<std::size_t>(input_offset)]));
                input_offset += run.stride;
            }
        });
        output_data[output_offset + output_linear] = static_cast<T>(accumulator);
    });
}

void run_sum(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
{
    assert(inputs.size() == 1);
    const auto &sum = dynamic_cast<const SumPrimitive &>(primitive);

    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        reduce_values<T>(inputs.front(), output, sum.axes(), T{},
                         [](T accumulator, T value) { return accumulator + value; });
    });
}

template <TensorElement T>
    requires std::floating_point<T>
void reduce_logsumexp(const TensorView &input, MutableTensorView output, std::span<const Shape::size_type> axes)
{
    assert(input.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));

    const ReductionPlan plan{input.shape(), input.layout(), axes};
    assert(plan.output_size() == output.shape().numel());
    if (plan.output_size() == 0)
    {
        return;
    }

    const T *input_data = data<T>(input);
    T *output_data = data<T>(output);
    const auto output_offset = static_cast<std::size_t>(output.layout().offset());

    const T negative_inf = -std::numeric_limits<T>::infinity();

    plan.for_each_output([&](Shape::size_type output_linear, const ReductionPlan::ReductionRange &range) {
        T maximum = negative_inf;
        bool contains_nan = false;

        // First pass to find the maximum
        range.for_each_run([&](const ReductionPlan::Run &run) {
            Layout::offset_type input_offset = run.offset;
            for (Shape::size_type i = 0; i < run.size; ++i)
            {
                assert(input_offset >= 0);
                const T value = input_data[static_cast<std::size_t>(input_offset)];
                if (std::isnan(value))
                {
                    contains_nan = true;
                }
                else
                {
                    maximum = std::max(maximum, value);
                }
                input_offset += run.stride;
            }
        });

        // Begin with the maximum and subtract the maximum from each input
        // before exponentiating and accumulating to avoid numerical instability.
        T result;

        if (contains_nan)
        {
            result = std::numeric_limits<T>::quiet_NaN();
        }
        else if (std::isinf(maximum))
        {
            result = maximum;
        }
        else
        {
            T exp_sum{};

            // Second pass to shift values, exponentiate, and sum
            range.for_each_run([&](const ReductionPlan::Run &run) {
                Layout::offset_type input_offset = run.offset;
                for (Shape::size_type i = 0; i < run.size; ++i)
                {
                    assert(input_offset >= 0);
                    const T value = input_data[static_cast<std::size_t>(input_offset)];
                    exp_sum += std::exp(value - maximum);
                    input_offset += run.stride;
                }
            });

            result = maximum + std::log(exp_sum);
        }

        output_data[output_offset + output_linear] = result;
    });
}

void run_logsumexp(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs,
                   MutableTensorView output)
{
    assert(inputs.size() == 1);
    const auto &logsumexp = dynamic_cast<const LogSumExpPrimitive &>(primitive);

    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        reduce_logsumexp<T>(inputs.front(), output, logsumexp.axes());
    });
}
} // namespace

void register_reduction_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{SumPrimitive::ID, DeviceType::Cpu}, run_sum);
    registry.register_kernel(KernelKey{LogSumExpPrimitive::ID, DeviceType::Cpu}, run_logsumexp);
}
} // namespace minitensor::detail::cpu
