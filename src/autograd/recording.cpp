#include "autograd/recording.hpp"

#include "autograd/node.hpp"
#include "autograd/tensor_access.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace minitensor::detail::autograd
{
    namespace
    {

        thread_local bool recording_enabled = true;

    } // namespace

    OperationContext::OperationContext(std::initializer_list<Tensor> parents)
        : parents_(parents),
          requires_grad_(
              recording_enabled &&
              std::ranges::any_of(
                  parents_,
                  [](const Tensor &parent)
                  { return parent.requires_grad(); }))
    {
    }

    bool OperationContext::requires_grad() const noexcept
    {
        return requires_grad_;
    }

    void OperationContext::record(
        Tensor &result,
        std::string name,
        BackwardFn backward,
        std::vector<Tensor> saved_tensors)
    {
        if (result.requires_grad() != requires_grad_)
        {
            throw std::logic_error(
                "operation result gradient state does not match its recording context");
        }
        if (!requires_grad_)
        {
            return;
        }
        if (!backward)
        {
            throw std::invalid_argument("cannot record an operation without a backward rule");
        }

        for (auto &saved_tensor : saved_tensors)
        {
            saved_tensor = saved_tensor.detach();
        }

        TensorAutogradAccess::set_grad_fn(
            result,
            std::make_unique<Node>(Node{
                std::move(name),
                std::move(parents_),
                std::move(saved_tensors),
                std::move(backward)}));
    }

    NoGradGuard::NoGradGuard() noexcept : previous_state_(recording_enabled)
    {
        recording_enabled = false;
    }

    NoGradGuard::~NoGradGuard()
    {
        recording_enabled = previous_state_;
    }

} // namespace minitensor::detail::autograd
