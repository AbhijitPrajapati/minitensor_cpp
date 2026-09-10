#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/execution/evaluator.hpp"
#include "tensor/execution/runtime_registry.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"
#include "../support/test_buffer.hpp"
#include "../support/test_primitive.hpp"
#include "../support/test_runtime.hpp"

namespace minitensor::test
{
    namespace
    {
        detail::ValueRef make_materialized_leaf(detail::ValueId id, const detail::TensorSpec &spec)
        {
            const auto size_bytes = spec.shape.numel() * dtype_size(spec.dtype);
            detail::ValueRef value = std::make_shared<detail::Value>(id, spec);
            value->materialize(detail::Materialization{
                make_test_buffer(size_bytes, spec.device),
                detail::Layout::contiguous(spec.shape)});
            return value;
        }

        detail::ValueRef apply_identity(const detail::ValueRef &input)
        {
            const std::array<detail::ValueRef, 1> inputs{input};
            return detail::apply_operation(std::make_unique<IdentitySpecPrimitive>(), inputs);
        }
    }

    void run_evaluator_test()
    {
        using detail::Buffer;
        using detail::Evaluator;
        using detail::KernelKey;
        using detail::KernelRegistry;
        using detail::Layout;
        using detail::MutableTensorView;
        using detail::Primitive;
        using detail::RuntimeRegistry;
        using detail::TensorSpec;
        using detail::TensorView;
        using detail::Value;
        using detail::ValueId;
        using detail::ValueRef;

        struct KernelCall final
        {
            detail::DeviceRuntime *runtime;
            const Primitive *primitive;
            const Buffer *input_buffer;
            Buffer *output_buffer;
            Shape output_shape;
            Layout output_layout;
        };

        const TensorSpec spec{Shape{2, 3}, DType::Float32, Device::cpu()};
        const ValueRef leaf = make_materialized_leaf(ValueId{500}, spec);
        const ValueRef intermediate = apply_identity(leaf);
        const ValueRef output = apply_identity(intermediate);

        RuntimeRegistry runtimes;
        auto runtime = std::make_unique<TestRuntime>();
        TestRuntime *const runtime_address = runtime.get();
        runtimes.register_runtime(std::move(runtime));

        std::vector<KernelCall> calls;
        KernelRegistry kernels;
        const KernelKey identity_key{
            typeid(IdentitySpecPrimitive), DeviceType::Cpu, DType::Float32};
        kernels.register_kernel(
            identity_key,
            [&calls](detail::DeviceRuntime &runtime,
                     const Primitive &primitive,
                     std::span<const TensorView> inputs,
                     MutableTensorView result)
            {
                expect(inputs.size() == 1, "the evaluator passes every node input to its kernel");
                calls.push_back(KernelCall{
                    &runtime,
                    &primitive,
                    &inputs.front().buffer(),
                    &result.buffer(),
                    result.shape(),
                    result.layout()});
            });

        Evaluator evaluator{runtimes, kernels};
        evaluator.evaluate(std::span<const ValueRef>{});
        expect(calls.empty() && runtime_address->allocation_sizes().empty(),
               "evaluating no roots performs no work");

        const std::array<ValueRef, 3> roots{intermediate, output, output};
        evaluator.evaluate(roots);

        expect(calls.size() == 2, "shared and duplicate graph values execute only once");
        expect(runtime_address->allocation_sizes().size() == 2,
               "the evaluator allocates one output buffer per executed value");
        expect(runtime_address->allocation_sizes()[0] == 6 * sizeof(float) &&
                   runtime_address->allocation_sizes()[1] == 6 * sizeof(float),
               "the evaluator allocates dense storage of the inferred byte size");
        expect(intermediate->materialization() != nullptr && output->materialization() != nullptr,
               "successful kernel execution materializes every planned output");

        expect(calls[0].runtime == runtime_address && calls[1].runtime == runtime_address,
               "the evaluator invokes kernels with the runtime selected for the output device");
        expect(calls[0].primitive == &intermediate->producer()->primitive() &&
                   calls[1].primitive == &output->producer()->primitive(),
               "each kernel receives the primitive belonging to its planned node");
        expect(calls[0].input_buffer == leaf->materialization()->buffer_ref().get() &&
                   calls[0].output_buffer == intermediate->materialization()->buffer_ref().get(),
               "the first kernel consumes the leaf and produces the intermediate value");
        expect(calls[1].input_buffer == intermediate->materialization()->buffer_ref().get() &&
                   calls[1].output_buffer == output->materialization()->buffer_ref().get(),
               "the evaluator executes dependent operations in topological order");
        expect(calls[0].output_shape == spec.shape && calls[1].output_shape == spec.shape,
               "kernel output views expose the inferred output shape");
        expect(calls[0].output_layout == Layout::contiguous(spec.shape) &&
                   calls[1].output_layout == Layout::contiguous(spec.shape),
               "the evaluator creates contiguous output materializations");

        evaluator.evaluate(std::array<ValueRef, 1>{output});
        expect(calls.size() == 2 && runtime_address->allocation_sizes().size() == 2,
               "evaluating an already materialized value reuses its cached result");

        {
            const ValueRef sharing_input = std::make_shared<Value>(ValueId{502}, spec);
            sharing_input->materialize(detail::Materialization{
                make_test_buffer(6 * sizeof(float)),
                Layout({1, 2})});

            const std::array<ValueRef, 1> sharing_inputs{sharing_input};
            const ValueRef sharing_output = detail::apply_operation(
                std::make_unique<IdentityStorageSharingPrimitive>(), sharing_inputs);
            expect(sharing_output->spec() == spec,
                   "an identity storage-sharing primitive preserves its input specification");

            RuntimeRegistry empty_runtimes;
            KernelRegistry empty_kernels;
            Evaluator sharing_evaluator{empty_runtimes, empty_kernels};
            sharing_evaluator.evaluate(std::array<ValueRef, 1>{sharing_output});

            const detail::Materialization *shared_materialization = sharing_output->materialization();
            expect(shared_materialization != nullptr,
                   "evaluation materializes a storage-sharing operation without a kernel or runtime");
            expect(shared_materialization->buffer_ref().get() ==
                       sharing_input->materialization()->buffer_ref().get(),
                   "a storage-sharing operation reuses its input buffer");
            expect(shared_materialization->layout() == Layout({1, 2}),
                   "a storage-sharing operation installs its primitive-derived layout");
        }

        const std::array<ValueRef, 1> null_roots{ValueRef{}};
        expect_throws<std::invalid_argument>(
            [&evaluator, &null_roots]
            {
                evaluator.evaluate(null_roots);
            },
            "evaluation rejects a null root value");

        const ValueRef unmaterialized_leaf = std::make_shared<Value>(ValueId{501}, spec);
        expect_throws<std::runtime_error>(
            [&evaluator, &unmaterialized_leaf]
            {
                evaluator.evaluate(std::array<ValueRef, 1>{unmaterialized_leaf});
            },
            "evaluation rejects an unmaterialized leaf");

        {
            RuntimeRegistry missing_runtime;
            KernelRegistry available_kernel;
            available_kernel.register_kernel(
                identity_key,
                [](detail::DeviceRuntime &,
                   const Primitive &,
                   std::span<const TensorView>,
                   MutableTensorView)
                {});
            Evaluator missing_runtime_evaluator{missing_runtime, available_kernel};
            const ValueRef unavailable_output = apply_identity(leaf);
            expect_throws<std::runtime_error>(
                [&missing_runtime_evaluator, &unavailable_output]
                {
                    missing_runtime_evaluator.evaluate(std::array<ValueRef, 1>{unavailable_output});
                },
                "evaluation rejects an output whose device has no runtime");
            expect(unavailable_output->materialization() == nullptr,
                   "missing runtime validation leaves the output unmaterialized");
        }

        {
            RuntimeRegistry available_runtime;
            auto tracking_runtime = std::make_unique<TestRuntime>();
            TestRuntime *const tracking_address = tracking_runtime.get();
            available_runtime.register_runtime(std::move(tracking_runtime));
            KernelRegistry missing_kernel;
            Evaluator missing_kernel_evaluator{available_runtime, missing_kernel};
            const ValueRef unavailable_output = apply_identity(leaf);
            expect_throws<std::runtime_error>(
                [&missing_kernel_evaluator, &unavailable_output]
                {
                    missing_kernel_evaluator.evaluate(std::array<ValueRef, 1>{unavailable_output});
                },
                "evaluation rejects an operation with no registered kernel");
            expect(tracking_address->allocation_sizes().empty() &&
                       unavailable_output->materialization() == nullptr,
                   "planning failure occurs before allocating or materializing output storage");
        }

        {
            RuntimeRegistry failing_runtime;
            auto tracking_runtime = std::make_unique<TestRuntime>();
            TestRuntime *const tracking_address = tracking_runtime.get();
            failing_runtime.register_runtime(std::move(tracking_runtime));
            KernelRegistry failing_kernel;
            failing_kernel.register_kernel(
                identity_key,
                [](detail::DeviceRuntime &,
                   const Primitive &,
                   std::span<const TensorView>,
                   MutableTensorView)
                {
                    throw std::runtime_error{"test kernel failure"};
                });
            Evaluator failing_evaluator{failing_runtime, failing_kernel};
            const ValueRef failed_output = apply_identity(leaf);
            expect_throws<std::runtime_error>(
                [&failing_evaluator, &failed_output]
                {
                    failing_evaluator.evaluate(std::array<ValueRef, 1>{failed_output});
                },
                "evaluation propagates kernel failures");
            expect(tracking_address->allocation_sizes().size() == 1,
                   "the evaluator allocates output storage before invoking its kernel");
            expect(failed_output->materialization() == nullptr,
                   "a kernel failure does not install a partial materialization");
        }
    }
}
