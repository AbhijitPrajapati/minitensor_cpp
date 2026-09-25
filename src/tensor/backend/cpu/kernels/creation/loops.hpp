#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/detail/random_engine.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
    template <CpuElement T>
    void fill(MutableTensorView output, T fill_value)
    {
        assert(output.dtype() == ElementDType<T>::value);

        const Shape &output_shape = output.shape();
        assert(output.layout().is_contiguous(output_shape));
        const std::size_t numel = output_shape.numel();
        if (numel == 0)
        {
            return;
        }

        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());
        std::fill_n(output_data + output_offset, numel, fill_value);
    }

    template <CpuElement T>
    void uniform(MutableTensorView output, T low, T high, RandomKey key)
    {
        assert(output.dtype() == ElementDType<T>::value);
        static_assert(std::is_floating_point_v<T>);

        const Shape &output_shape = output.shape();
        assert(output.layout().is_contiguous(output_shape));
        const std::size_t numel = output_shape.numel();
        if (numel == 0)
        {
            return;
        }

        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        if (low == high)
        {
            std::fill_n(output_data + output_offset, numel, low);
            return;
        }

        const auto low_double = static_cast<double>(low);
        const auto range = static_cast<double>(high) - low_double;
        const T largest_valid = std::nextafter(high, low);
        std::size_t base = 0;
        std::uint64_t block_idx = 0;
        
        while (base < numel)
        {
            const RandomBlock block = random_block(key, block_idx);
            const std::size_t count =
                std::min<std::size_t>(block.words.size(), numel - base);
            for (std::size_t lane = 0; lane < count; ++lane)
            {
                const double unit = static_cast<double>(unit_uniform(block.words[lane]));
                T value = static_cast<T>(low_double + range * unit);
                // prevent rounding from breaking contract
                if (value >= high)
                {
                    value = largest_valid;
                }
                output_data[output_offset + base + lane] = value;
            }
            base += count;
            ++block_idx;
        }
    }

    template <CpuElement T>
    void normal(MutableTensorView output, T mean, T std_dev, RandomKey key)
    {
        assert(output.dtype() == ElementDType<T>::value);
        static_assert(std::is_floating_point_v<T>);

        const Shape &output_shape = output.shape();
        assert(output.layout().is_contiguous(output_shape));
        const std::size_t numel = output_shape.numel();
        if (numel == 0)
        {
            return;
        }

        T *output_data = data<T>(output);
        const auto output_offset = static_cast<std::size_t>(output.layout().offset());

        if (std_dev == T{0})
        {
            std::fill_n(output_data + output_offset, numel, mean);
            return;
        }

        const auto mean_double = static_cast<double>(mean);
        const auto std_dev_double = static_cast<double>(std_dev);
        constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
        std::size_t base = 0;
        std::uint64_t block_idx = 0;

        while (base < numel)
        {
            const RandomBlock block = random_block(key, block_idx);
            std::array<double, 4> standard_normals{};
            for (std::size_t pair = 0; pair < 2; ++pair)
            {
                const double u1 = open_unit_uniform(block.words[pair * 2]);
                const double u2 = open_unit_uniform(block.words[pair * 2 + 1]);
                const double radius = std::sqrt(-2.0 * std::log(u1));
                const double angle = two_pi * u2;
                standard_normals[pair * 2] = radius * std::cos(angle);
                standard_normals[pair * 2 + 1] = radius * std::sin(angle);
            }

            const std::size_t count =
                std::min<std::size_t>(standard_normals.size(), numel - base);
            for (std::size_t lane = 0; lane < count; ++lane)
            {
                output_data[output_offset + base + lane] = static_cast<T>(
                    mean_double + std_dev_double * standard_normals[lane]);
            }

            base += count;
            ++block_idx;
        }
    }
}
