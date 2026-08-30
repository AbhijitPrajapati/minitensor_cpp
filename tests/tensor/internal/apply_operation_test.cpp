#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <variant>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/origin.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/ops/full.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_apply_operation_test()
    {
        using detail::FullPrimitive;
        using detail::LeafOrigin;
        using detail::PrimitiveRef;
        using detail::ProducedOrigin;
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueRef;

        const TensorSpec input_spec{Shape{2, 3}, DType::Float32, Device::cpu(2)};
        const ValueRef input = std::make_shared<Value>(detail::next_value_id(), input_spec, LeafOrigin{});
        const auto primitive = std::make_shared<IdentitySpecPrimitive>();
        const std::array<ValueRef, 1> inputs{input};

        const ValueRef output = detail::apply_operation(primitive, inputs);
        expect(output != nullptr, "apply_operation returns an output value");
        expect(output.get() != input.get(), "apply_operation creates a distinct output value");
        expect(output->id() != input->id(), "apply_operation assigns a new value id");
        expect(output->spec() == input_spec, "apply_operation uses the primitive's inferred tensor specification");
        expect(output->materialization() == nullptr, "apply_operation constructs a lazy, unmaterialized output");

        const ProducedOrigin *origin = std::get_if<ProducedOrigin>(&output->origin());
        expect(origin != nullptr, "apply_operation marks its output as produced");
        expect(origin->node != nullptr, "apply_operation attaches a producer node");
        expect(origin->node->primitive_ref().get() == primitive.get(),
               "the producer node owns the applied primitive");
        expect(origin->node->inputs().size() == 1 && origin->node->inputs().front().get() == input.get(),
               "the producer node owns the applied inputs in order");

        expect_throws<std::invalid_argument>(
            [&inputs]
            {
                (void)detail::apply_operation(PrimitiveRef{}, inputs);
            },
            "apply_operation rejects a null primitive");

        const std::array<ValueRef, 1> null_inputs{ValueRef{}};
        expect_throws<std::invalid_argument>(
            [&primitive, &null_inputs]
            {
                (void)detail::apply_operation(primitive, null_inputs);
            },
            "apply_operation rejects null inputs");
        expect_throws<std::invalid_argument>(
            [&primitive]
            {
                (void)detail::apply_operation(primitive, std::span<const ValueRef>{});
            },
            "apply_operation propagates primitive input-count validation");

        const TensorSpec generated_spec{Shape{4, 1}, DType::Float32, Device::cpu(3)};
        const auto full_primitive = std::make_shared<FullPrimitive>(generated_spec, 7.25F);
        const ValueRef generated = detail::apply_operation(full_primitive, std::span<const ValueRef>{});
        expect(generated->spec() == generated_spec, "a zero-input primitive can infer a fixed output spec");

        const ProducedOrigin &generated_origin = std::get<ProducedOrigin>(generated->origin());
        const auto *stored_full = dynamic_cast<const FullPrimitive *>(&generated_origin.node->primitive());
        expect(stored_full != nullptr, "the producer retains the concrete primitive type");
        expect(stored_full->name() == "full", "the full primitive reports its operation name");
        expect(stored_full->fill_value() == 7.25F, "the full primitive retains its fill value");

        expect_throws<std::invalid_argument>(
            [&full_primitive, &inputs]
            {
                (void)detail::apply_operation(full_primitive, inputs);
            },
            "a zero-input primitive rejects unexpected inputs during inference");
    }
}
