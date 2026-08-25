#pragma once

#include "minitensor/tensor.hpp"

#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace minitensor::detail::autograd
{

    using GradList = std::vector<std::optional<Tensor>>;
    using TensorSpan = std::span<const Tensor>;
    using BackwardFn = std::function<GradList(
        const Tensor &grad_output,
        TensorSpan parents,
        TensorSpan saved_tensors)>;

    // Describes one potential graph edge before the forward result is created.
    // The context is the single owner of the parent ordering used by both graph
    // traversal and the backward rule.
    class OperationContext final
    {
    public:
        OperationContext(std::initializer_list<Tensor> parents);

        [[nodiscard]] bool requires_grad() const noexcept;
        void record(
            Tensor &result,
            std::string name,
            BackwardFn backward,
            std::vector<Tensor> saved_tensors = {});

    private:
        std::vector<Tensor> parents_;
        bool requires_grad_;
    };

    // Backward rules use ordinary tensor operations for their numerical work,
    // but this first-order engine deliberately does not record those operations.
    class NoGradGuard final
    {
    public:
        NoGradGuard() noexcept;
        ~NoGradGuard();

        NoGradGuard(const NoGradGuard &) = delete;
        NoGradGuard &operator=(const NoGradGuard &) = delete;

    private:
        bool previous_state_;
    };

} // namespace minitensor::detail::autograd
