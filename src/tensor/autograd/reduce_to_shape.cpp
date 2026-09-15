#include "reduce_to_shape.hpp"

#include <stdexcept>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>


namespace minitensor::detail
{
    [[nodiscard]] Tensor reduce_to_shape(const Tensor &cotangent, const Shape &target_shape)
    {
        const Shape &cotangent_shape = cotangent.shape();

        if (cotangent_shape == target_shape)
        {
            return cotangent;
        }

        if (target_shape.rank() > cotangent_shape.rank())
        {
            throw std::logic_error{"cannot reduce a tensor to a higher rank shape"};
        }
    }
}