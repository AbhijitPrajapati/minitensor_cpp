#include "autograd/engine.hpp"

#include "autograd/node.hpp"
#include "autograd/recording.hpp"
#include "autograd/tensor_access.hpp"
#include "minitensor/ops.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minitensor::detail::autograd
{
    namespace
    {

        using GradientMap = std::unordered_map<TensorId, Tensor, TensorIdHash>;

        void visit(
            const Tensor &tensor,
            std::unordered_set<TensorId, TensorIdHash> &visited,
            std::vector<Tensor> &postorder)
        {
            const auto key = TensorAutogradAccess::identity(tensor);
            if (!visited.insert(key).second)
            {
                return;
            }

            const auto *node = TensorAutogradAccess::grad_fn(tensor);
            if (node)
            {
                for (const auto &parent : node->parents)
                {
                    if (parent.requires_grad())
                    {
                        visit(parent, visited, postorder);
                    }
                }
            }
            postorder.push_back(tensor);
        }

        void accumulate_pending(
            GradientMap &pending,
            const Tensor &tensor,
            const Tensor &contribution)
        {
            const auto key = TensorAutogradAccess::identity(tensor);
            const auto detached = contribution.detach();
            const auto found = pending.find(key);
            if (found == pending.end())
            {
                pending.emplace(key, detached);
            }
            else
            {
                found->second = found->second + detached;
            }
        }

        void accumulate_leaf(Tensor &leaf, const Tensor &contribution)
        {
            const auto existing = leaf.grad();
            if (!existing.has_value())
            {
                // Publicly visible gradients use a predictable dense layout even when
                // a view backward function produced a strided gradient.
                TensorAutogradAccess::set_grad(
                    leaf, contribution.contiguous().detach());
                return;
            }

            const auto accumulated = *existing + contribution.detach();
            TensorAutogradAccess::set_grad(leaf, accumulated);
        }

    } // namespace

    void run_backward(const Tensor &root, const Tensor &gradient)
    {
        if (!root.requires_grad())
        {
            throw std::logic_error("cannot call backward on a tensor that does not require gradients");
        }
        if (root.shape() != gradient.shape())
        {
            throw std::invalid_argument("backward gradient shape must match the output shape");
        }

        std::unordered_set<TensorId, TensorIdHash> visited;
        std::vector<Tensor> postorder;
        visit(root, visited, postorder);

        GradientMap pending;
        accumulate_pending(pending, root, gradient);

        const NoGradGuard no_grad;

        for (auto current_iterator = postorder.rbegin();
             current_iterator != postorder.rend();
             ++current_iterator)
        {
            auto current = *current_iterator;
            const auto key = TensorAutogradAccess::identity(current);
            const auto pending_gradient = pending.find(key);
            if (pending_gradient == pending.end())
            {
                continue;
            }

            const auto current_gradient = pending_gradient->second;
            const auto *node = TensorAutogradAccess::grad_fn(current);
            if (!node)
            {
                accumulate_leaf(current, current_gradient);
                continue;
            }

            const auto parent_gradients = node->backward(
                current_gradient,
                TensorSpan{node->parents},
                TensorSpan{node->saved_tensors});
            if (parent_gradients.size() != node->parents.size())
            {
                throw std::logic_error(
                    "backward function for " + node->name +
                    " returned the wrong number of gradients");
            }

            for (std::size_t index = 0; index < node->parents.size(); ++index)
            {
                const auto &parent = node->parents[index];
                if (!parent.requires_grad())
                {
                    continue;
                }
                if (!parent_gradients[index].has_value())
                {
                    throw std::logic_error(
                        "backward function for " + node->name +
                        " omitted a required parent gradient");
                }
                if (parent_gradients[index]->shape() != parent.shape())
                {
                    throw std::logic_error(
                        "backward function for " + node->name +
                        " produced a gradient with the wrong shape");
                }
                accumulate_pending(pending, parent, *parent_gradients[index]);
            }
        }
    }

} // namespace minitensor::detail::autograd
