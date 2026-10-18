#include "dense_size.hpp"

#include <limits>
#include <stdexcept>

#include <minitensor/types.hpp>

#include "tensor_spec.hpp"

namespace minitensor::detail
{
// This function computes the buffer size in bytes for a given tensor specification.
std::size_t dense_size_bytes(const TensorSpec &spec)
{
    const std::size_t element_size = dtype_size(spec.dtype);
    const std::size_t numel = spec.shape.numel();
    if (numel > std::numeric_limits<std::size_t>::max() / element_size)
    {
        throw std::overflow_error{"tensor storage size exceeds size_t"};
    }
    return numel * element_size;
}
} // namespace minitensor::detail