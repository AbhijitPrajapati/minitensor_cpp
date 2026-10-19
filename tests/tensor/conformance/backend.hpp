#pragma once

#include <initializer_list>
#include <ranges>
#include <string_view>
#include <utility>

#include <minitensor/data.hpp>
#include <minitensor/host_scalar.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::test::conformance
{
struct Backend final
{
    std::string_view name;
    Device device;
};

[[nodiscard]] inline Backend backend() noexcept
{
#if defined(MINITENSOR_TEST_CONFORMANCE_CPU)
    return Backend{"cpu", Device::cpu()};
#else
#error "A conformance backend must be selected by the test target"
#endif
}

struct FromData final
{
    template <TensorElement T>
    [[nodiscard]] Tensor operator()(std::initializer_list<T> values, Shape shape) const
    {
        return minitensor::from_data<T>(values, std::move(shape), backend().device);
    }

    template <std::ranges::contiguous_range Range>
        requires std::ranges::sized_range<const Range &> && TensorElement<std::ranges::range_value_t<Range>>
    [[nodiscard]] Tensor operator()(const Range &values, Shape shape) const
    {
        return minitensor::from_data(values, std::move(shape), backend().device);
    }
};

struct Full final
{
    [[nodiscard]] Tensor operator()(Shape shape, HostScalar value) const
    {
        return minitensor::full(std::move(shape), value, TensorOptions{DType::Float32, backend().device});
    }
};

inline constexpr FromData from_data;
inline constexpr Full full;
} // namespace minitensor::test::conformance
