#include <minitensor/evaluation.hpp>

#include <vector>
#include <array>

#include "tensor_access.hpp"
#include "tensor/execution/environment.hpp"
#include "tensor/graph/fwd.hpp"

namespace minitensor
{
    void eval(std::span<const Tensor> tensors)
    {
        std::vector<detail::ValueRef> roots;
        roots.reserve(tensors.size());
        for (const Tensor &tensor : tensors)
        {
            roots.push_back(detail::TensorAccess::value(tensor));
        }
        detail::environment().evaluate(roots);
    }

    void eval(const Tensor &tensor)
    {
        const std::array<Tensor, 1> tensors{tensor};
        eval(tensors);
    }
}
