#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace minitensor::detail::cuda
{
inline void cuda_check(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline bool cuda_report(cudaError_t status, const char *operation) noexcept
{
    if (status == cudaSuccess)
    {
        return true;
    }

    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}
} // namespace minitensor::detail::cuda