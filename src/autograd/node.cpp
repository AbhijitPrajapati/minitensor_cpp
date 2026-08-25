#include "autograd/node.hpp"

#include "autograd/tensor_access.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace minitensor::detail::autograd
{

    void set_history(
        Tensor &result,
        std::string name,
        std::vector<Tensor> parents,
        BackwardFn backward)
    {
        if (!result.requires_grad())
        {
            throw std::logic_error("cannot attach autograd history to a tensor that does not require gradients");
        }
        TensorAutogradAccess::set_grad_fn(
            result,
            std::make_unique<Node>(Node{
                std::move(name), std::move(parents), std::move(backward)}));
    }

} // namespace minitensor::detail::autograd
