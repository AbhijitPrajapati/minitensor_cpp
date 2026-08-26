#include "autograd/node.hpp"
#include "autograd/recording.hpp"
#include "autograd/tensor_access.hpp"

#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

    using minitensor::Shape;
    using minitensor::Tensor;
    using minitensor::detail::autograd::GradList;
    using minitensor::detail::autograd::NoGradGuard;
    using minitensor::detail::autograd::OperationContext;
    using minitensor::detail::autograd::TensorAutogradAccess;

    int failures = 0;

    void expect(const bool condition, const std::string &message)
    {
        if (!condition)
        {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Exception>
    void expect_throws(const std::function<void()> &action, const std::string &message)
    {
        try
        {
            action();
            expect(false, message + " (no exception was thrown)");
        }
        catch (const Exception &)
        {
        }
        catch (...)
        {
            expect(false, message + " (wrong exception type)");
        }
    }

} // namespace

int main()
{
    auto parent = Tensor::ones(Shape{2}, true);
    OperationContext context{parent};
    expect(context.requires_grad(), "recording context derives gradient state from parents");

    auto result = Tensor::zeros(Shape{2}, context.requires_grad());
    bool backward_called = false;
    context.record(
        result,
        "test_operation",
        [&backward_called](
            const Tensor &gradient,
            const minitensor::detail::autograd::TensorSpan parents,
            const minitensor::detail::autograd::TensorSpan saved_tensors)
        {
            backward_called = true;
            expect(parents.size() == 1, "backward rule receives the node parent list");
            expect(saved_tensors.size() == 1, "backward rule receives explicit saved tensors");
            return GradList{std::optional<Tensor>{gradient}};
        },
        {result});

    const auto *node = TensorAutogradAccess::grad_fn(result);
    expect(node != nullptr, "recording context installs a graph node");
    if (node)
    {
        expect(node->name == "test_operation", "recorded node retains its operation name");
        expect(node->parents.size() == 1, "recorded node owns one authoritative parent list");
        expect(node->saved_tensors.size() == 1, "recorded node owns explicit saved tensors");
        expect(
            TensorAutogradAccess::identity(node->parents[0]) ==
                TensorAutogradAccess::identity(parent),
            "recorded parent identity matches the forward operand");
        expect(
            !node->saved_tensors[0].requires_grad(),
            "recording context detaches saved tensors");
    }

    result.backward(Tensor::ones(Shape{2}));
    expect(backward_called, "engine invokes the recorded backward rule");
    expect(parent.grad().has_value(), "recorded parent receives its gradient");
    expect(
        parent.grad().has_value() && !parent.grad()->requires_grad(),
        "first-order backward leaves accumulated gradients untracked");

    {
        const NoGradGuard no_grad;
        OperationContext suppressed{parent};
        expect(
            !suppressed.requires_grad(),
            "no-grad guard suppresses recording for differentiable parents");
        auto untracked = Tensor::zeros(Shape{2}, suppressed.requires_grad());
        suppressed.record(untracked, "suppressed", {});
        expect(
            TensorAutogradAccess::grad_fn(untracked) == nullptr,
            "suppressed operations do not install graph nodes");
    }

    OperationContext restored{parent};
    expect(restored.requires_grad(), "no-grad guard restores the previous recording state");
    expect_throws<std::logic_error>(
        [&]
        {
            auto mismatched = Tensor::zeros(Shape{2});
            restored.record(mismatched, "mismatched", {});
        },
        "recording context rejects an inconsistent result gradient state");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All autograd contract tests passed\n";
    return 0;
}
