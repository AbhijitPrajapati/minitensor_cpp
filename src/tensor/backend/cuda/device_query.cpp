#include "device_query.hpp"

#include <cstdint>
#include <cuda_runtime_api.h>

#include "detail/cuda_check.hpp"

namespace minitensor::detail::cuda
{
std::uint32_t cuda_device_count()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver)
    {
        return 0;
    }
    cuda_check(err, "query CUDA device count");
    return static_cast<std::uint32_t>(count);
}
} // namespace minitensor::detail::cuda