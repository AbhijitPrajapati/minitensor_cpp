#pragma once

#include <initializer_list>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
// Data transfer functions

template <TensorElement T = float>
[[nodiscard]] std::vector<T> to_vector(const Tensor &tensor);

template <TensorElement T = float>
[[nodiscard]] T item(const Tensor &tensor);

template <TensorElement T>
[[nodiscard]] Tensor from_data(std::span<const T> data, Shape shape, Device device = Device::cpu());

template <TensorElement T>
[[nodiscard]] Tensor from_data(std::initializer_list<T> data, Shape shape, Device device = Device::cpu())
{
    return from_data<T>(std::span<const T>(data.begin(), data.size()), std::move(shape), device);
}

// In addition to span and initializer list overloads, a contiguous range
// overload is required for from_data because the compiler cannot deduce
// T from say, std::array<float, N>. This function simply extracts T
// and calls the span overload.
template <typename Range>
    requires(std::ranges::contiguous_range<const Range &> && std::ranges::sized_range<const Range &> &&
             TensorElement<std::ranges::range_value_t<Range>>)
[[nodiscard]] Tensor from_data(const Range &data, Shape shape, Device device = Device::cpu())
{
    using T = std::ranges::range_value_t<Range>;
    return from_data<T>(std::span<const T>(std::ranges::data(data), std::ranges::size(data)), std::move(shape), device);
}

// Templated function definitions must be visible to all compilation units
// unless an external definition is specified.
extern template std::vector<float> to_vector<float>(const Tensor &tensor);
extern template std::vector<std::int64_t> to_vector<std::int64_t>(const Tensor &tensor);
extern template float item<float>(const Tensor &tensor);
extern template std::int64_t item<std::int64_t>(const Tensor &tensor);
extern template Tensor from_data<float>(std::span<const float> data, Shape shape, Device device);
extern template Tensor from_data<std::int64_t>(std::span<const std::int64_t> data, Shape shape, Device device);
} // namespace minitensor
