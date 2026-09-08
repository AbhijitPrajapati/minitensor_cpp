#include "dense_size.hpp"

#include <minitensor/types.hpp>

#include "tensor_spec.hpp"
#include "checked_arithmetic.hpp"

namespace minitensor::detail
{
    std::size_t dense_size_bytes(const TensorSpec &spec)
    {
        const std::size_t element_size = dtype_size(spec.dtype);
        const std::size_t numel = spec.shape.numel();
        return checked_multiply(numel, element_size);
    }
}