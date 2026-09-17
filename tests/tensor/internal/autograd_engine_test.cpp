#include <minitensor/data.hpp>
#include <minitensor/ops.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/autograd/engine.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/tensor_access.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    namespace
    {
        Tensor apply_vjp_rule(
            std::span<const Tensor> inputs,
            detail::TensorSpec output_spec,
            RuleBasedVjpPrimitive::Rule rule)
        {
            std::vector<detail::ValueRef> input_values;
            input_values.reserve(inputs.size());
            for (const Tensor &input : inputs)
            {
                input_values.push_back(detail::TensorAccess::value(input));
            }

            detail::ValueRef output = detail::apply_operation(
                std::make_unique<RuleBasedVjpPrimitive>(
                    inputs.size(), std::move(output_spec), std::move(rule)),
                input_values);
            return detail::TensorAccess::make(std::move(output));
        }
    }

    void run_autograd_engine_test()
    {
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueRef;

        const TensorSpec spec{Shape{2}, DType::Float32, Device::cpu()};
        const Tensor source = full(spec.shape, 3.0F);
        const std::array<float, 2> seed_values{1.5F, -2.0F};
        const Tensor seed = from_data(seed_values, spec.shape);

        std::vector<int> call_order;
        const Value *first_output_value = nullptr;
        const std::array<Tensor, 1> first_inputs{source};
        const Tensor first = apply_vjp_rule(
            first_inputs,
            spec,
            [&call_order, &source, &seed, &first_output_value](
                std::span<const Tensor> inputs,
                const Tensor &output,
                const Tensor &output_cotangent)
            {
                call_order.push_back(1);
                expect(inputs.size() == 1 &&
                           detail::TensorAccess::value(inputs.front()).get() ==
                               detail::TensorAccess::value(source).get(),
                       "a VJP rule receives its node inputs in order");
                expect(detail::TensorAccess::value(output).get() == first_output_value,
                       "a VJP rule receives its node output");
                expect(detail::TensorAccess::value(output_cotangent).get() ==
                           detail::TensorAccess::value(seed).get(),
                       "a VJP rule receives the cotangent propagated from its consumer");
                return std::vector<std::optional<Tensor>>{output_cotangent};
            });
        first_output_value = detail::TensorAccess::value(first).get();

        const Value *final_output_value = nullptr;
        const std::array<Tensor, 1> final_inputs{first};
        const Tensor output = apply_vjp_rule(
            final_inputs,
            spec,
            [&call_order, &first, &seed, &final_output_value](
                std::span<const Tensor> inputs,
                const Tensor &node_output,
                const Tensor &output_cotangent)
            {
                call_order.push_back(2);
                expect(inputs.size() == 1 &&
                           detail::TensorAccess::value(inputs.front()).get() ==
                               detail::TensorAccess::value(first).get(),
                       "a downstream VJP rule receives its immediate graph input");
                expect(detail::TensorAccess::value(node_output).get() == final_output_value,
                       "a downstream VJP rule receives its own node output");
                expect(detail::TensorAccess::value(output_cotangent).get() ==
                           detail::TensorAccess::value(seed).get(),
                       "the reverse pass starts with the supplied output cotangent");
                return std::vector<std::optional<Tensor>>{output_cotangent};
            });
        final_output_value = detail::TensorAccess::value(output).get();

        const std::vector<Tensor> no_targets = detail::reverse_vjp(
            detail::TensorAccess::value(output),
            std::span<const ValueRef>{},
            seed);
        expect(no_targets.empty() && call_order.empty(),
               "reverse_vjp performs no traversal when no targets are requested");

        const std::array<ValueRef, 1> source_target{
            detail::TensorAccess::value(source)};
        const std::vector<Tensor> chain_cotangents = detail::reverse_vjp(
            detail::TensorAccess::value(output), source_target, seed);
        expect(call_order == std::vector<int>{2, 1},
               "reverse_vjp invokes primitive rules in reverse topological order");
        expect(chain_cotangents.size() == 1 &&
                   detail::TensorAccess::value(chain_cotangents.front()).get() ==
                       detail::TensorAccess::value(seed).get(),
               "reverse_vjp returns the cotangent propagated to a target");

        std::size_t accumulation_calls = 0;
        const std::array<Tensor, 2> duplicate_inputs{source, source};
        const Tensor duplicate_output = apply_vjp_rule(
            duplicate_inputs,
            spec,
            [&accumulation_calls](
                std::span<const Tensor> inputs,
                const Tensor &,
                const Tensor &output_cotangent)
            {
                ++accumulation_calls;
                expect(inputs.size() == 2,
                       "a VJP rule receives duplicate graph inputs separately");
                return std::vector<std::optional<Tensor>>{
                    output_cotangent, output_cotangent};
            });
        const std::vector<Tensor> accumulated = detail::reverse_vjp(
            detail::TensorAccess::value(duplicate_output), source_target, seed);
        const std::array<float, 2> expected_accumulated{3.0F, -4.0F};
        expect(accumulation_calls == 1 && accumulated.size() == 1 &&
                   std::ranges::equal(to_vector(accumulated.front()), expected_accumulated),
               "reverse_vjp adds multiple cotangent contributions to the same target");

        std::size_t unrelated_calls = 0;
        const Tensor unrelated_source = full(spec.shape, 9.0F);
        const std::array<Tensor, 1> unrelated_inputs{unrelated_source};
        const Tensor unrelated_branch = apply_vjp_rule(
            unrelated_inputs,
            spec,
            [&unrelated_calls](
                std::span<const Tensor>, const Tensor &, const Tensor &output_cotangent)
            {
                ++unrelated_calls;
                return std::vector<std::optional<Tensor>>{output_cotangent};
            });

        std::size_t join_calls = 0;
        const std::array<Tensor, 2> join_inputs{source, unrelated_branch};
        const Tensor joined = apply_vjp_rule(
            join_inputs,
            spec,
            [&join_calls](
                std::span<const Tensor>, const Tensor &, const Tensor &output_cotangent)
            {
                ++join_calls;
                return std::vector<std::optional<Tensor>>{
                    output_cotangent, output_cotangent};
            });
        const std::vector<Tensor> pruned = detail::reverse_vjp(
            detail::TensorAccess::value(joined), source_target, seed);
        expect(join_calls == 1 && unrelated_calls == 0 && pruned.size() == 1,
               "reverse_vjp prunes branches that cannot reach a requested target");

        const std::size_t calls_before_blocked_path = call_order.size();
        const std::array<Tensor, 1> blocked_inputs{first};
        const Tensor blocked = apply_vjp_rule(
            blocked_inputs,
            spec,
            [](std::span<const Tensor>, const Tensor &, const Tensor &)
            {
                return std::vector<std::optional<Tensor>>{std::nullopt};
            });
        const std::vector<Tensor> blocked_cotangents = detail::reverse_vjp(
            detail::TensorAccess::value(blocked), source_target, seed);
        const std::array<float, 2> expected_zeroes{};
        expect(call_order.size() == calls_before_blocked_path &&
                   blocked_cotangents.size() == 1 &&
                   std::ranges::equal(to_vector(blocked_cotangents.front()), expected_zeroes),
               "a missing primitive contribution skips earlier VJP steps and yields a zero target cotangent");

        expect_throws<std::invalid_argument>(
            [&seed]
            {
                (void)detail::reverse_vjp(
                    ValueRef{}, std::span<const ValueRef>{}, seed);
            },
            "reverse_vjp rejects a null output value");

        const std::array<ValueRef, 1> null_targets{ValueRef{}};
        expect_throws<std::invalid_argument>(
            [&output, &null_targets, &seed]
            {
                (void)detail::reverse_vjp(
                    detail::TensorAccess::value(output), null_targets, seed);
            },
            "reverse_vjp rejects a null target value");

        const Tensor wrong_count = apply_vjp_rule(
            blocked_inputs,
            spec,
            [](std::span<const Tensor>, const Tensor &, const Tensor &)
            {
                return std::vector<std::optional<Tensor>>{};
            });
        expect_throws<std::logic_error>(
            [&wrong_count, &source_target, &seed]
            {
                (void)detail::reverse_vjp(
                    detail::TensorAccess::value(wrong_count), source_target, seed);
            },
            "reverse_vjp rejects a primitive contribution count that differs from its input count");

        const Tensor wrong_spec = apply_vjp_rule(
            blocked_inputs,
            spec,
            [](std::span<const Tensor>, const Tensor &, const Tensor &)
            {
                return std::vector<std::optional<Tensor>>{
                    full(Shape{1}, 1.0F)};
            });
        expect_throws<std::logic_error>(
            [&wrong_spec, &source_target, &seed]
            {
                (void)detail::reverse_vjp(
                    detail::TensorAccess::value(wrong_spec), source_target, seed);
            },
            "reverse_vjp rejects a primitive contribution with the wrong tensor specification");

        const std::array<ValueRef, 1> default_inputs{
            detail::TensorAccess::value(source)};
        const ValueRef default_output = detail::apply_operation(
            std::make_unique<IdentitySpecPrimitive>(), default_inputs);
        expect_throws<std::logic_error>(
            [&default_output, &source_target, &seed]
            {
                (void)detail::reverse_vjp(default_output, source_target, seed);
            },
            "reverse_vjp propagates a primitive's unimplemented-VJP error");
    }
}
