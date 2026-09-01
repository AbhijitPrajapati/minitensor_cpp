#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/ops/full.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_apply_operation_test()
    {
        using detail::FullPrimitive;
        using detail::Primitive;
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueRef;

        const TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cpu(2)};
        const ValueRef input = std::make_shared<Value>(detail::next_value_id(), input_spec);
        auto primitive = std::make_unique<IdentitySpecPrimitive>();
        const Primitive *primitive_address = primitive.get();
        const std::array<ValueRef, 1> inputs{input};

        const ValueRef output = detail::apply_operation(std::move(primitive), inputs);
        expect(output != nullptr, "apply_operation returns an output value");
        expect(primitive == nullptr, "apply_operation takes ownership of its primitive");
        expect(output.get() != input.get(), "apply_operation creates a distinct output value");
        expect(output->id() != input->id(), "apply_operation assigns a new value id");
        expect(output->spec() == input_spec, "apply_operation uses the primitive's inferred tensor specification");
        expect(output->materialization() == nullptr, "apply_operation constructs a lazy, unmaterialized output");

        expect(!output->is_leaf(), "apply_operation marks its output as produced");
        const auto *producer = output->producer();
        expect(producer != nullptr, "apply_operation attaches a producer node");
        expect(&producer->primitive() == primitive_address,
               "the producer node owns the applied primitive");
        expect(producer->inputs().size() == 1 && producer->inputs().front().get() == input.get(),
               "the producer node owns the applied inputs in order");

        expect_throws<std::invalid_argument>(
            [&inputs]
            {
                (void)detail::apply_operation(std::unique_ptr<Primitive>{}, inputs);
            },
            "apply_operation rejects a null primitive");

        const std::array<ValueRef, 1> null_inputs{ValueRef{}};
        expect_throws<std::invalid_argument>(
            [&null_inputs]
            {
                (void)detail::apply_operation(std::make_unique<IdentitySpecPrimitive>(), null_inputs);
            },
            "apply_operation rejects null inputs");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)detail::apply_operation(std::make_unique<IdentitySpecPrimitive>(),
                                              std::span<const ValueRef>{});
            },
            "apply_operation propagates primitive input-count validation");

        const TensorSpec generated_spec{Shape{4, 1}, DType::Float32, Device::cpu(3)};
        auto full_primitive = std::make_unique<FullPrimitive>(generated_spec, 7.25F);
        const Primitive *full_primitive_address = full_primitive.get();
        const ValueRef generated = detail::apply_operation(std::move(full_primitive), std::span<const ValueRef>{});
        expect(generated->spec() == generated_spec, "a zero-input primitive can infer a fixed output spec");
        expect(full_primitive == nullptr, "apply_operation transfers ownership of a zero-input primitive");

        const auto *generated_producer = generated->producer();
        expect(generated_producer != nullptr, "a zero-input primitive attaches a producer node");
        const auto *stored_full = dynamic_cast<const FullPrimitive *>(&generated_producer->primitive());
        expect(stored_full != nullptr, "the producer retains the concrete primitive type");
        expect(stored_full == full_primitive_address, "the producer retains the supplied primitive object");
        expect(stored_full->name() == "full", "the full primitive reports its operation name");
        expect(stored_full->fill_value() == 7.25F, "the full primitive retains its fill value");

        expect_throws<std::invalid_argument>(
            [&generated_spec, &inputs]
            {
                (void)detail::apply_operation(std::make_unique<FullPrimitive>(generated_spec, 7.25F), inputs);
            },
            "a zero-input primitive rejects unexpected inputs during inference");
    }
}
