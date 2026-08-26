#include "core/layout.hpp"
#include "core/shape.hpp"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

    using minitensor::Coordinates;
    using minitensor::Shape;
    using minitensor::Strides;
    using minitensor::detail::Layout;
    using minitensor::detail::shape::broadcast;
    using minitensor::detail::shape::coordinates_from_linear;
    using minitensor::detail::shape::is_broadcastable_to;
    using minitensor::detail::shape::numel;
    using minitensor::detail::shape::reduce;
    using minitensor::detail::shape::require_reshape_compatible;

    int failures = 0;

    void expect(const bool condition, const std::string &message)
    {
        if (!condition)
        {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Exception>
    void expect_throws(const std::function<void()> &action, const std::string &message)
    {
        try
        {
            action();
            expect(false, message + " (no exception was thrown)");
        }
        catch (const Exception &)
        {
        }
        catch (...)
        {
            expect(false, message + " (wrong exception type)");
        }
    }

} // namespace

int main()
{
    const Shape matrix_shape{2, 3};
    expect(numel(matrix_shape) == 6, "shape element count is correct");
    expect(numel(Shape{}) == 1, "rank-zero shape represents one scalar element");
    expect(
        broadcast(Shape{2, 1, 3}, Shape{4, 3}) == Shape{2, 4, 3},
        "shape infers a trailing-dimension broadcast result");
    expect(
        is_broadcastable_to(Shape{3}, Shape{2, 3}), "shape recognizes a valid broadcast target");
    expect(
        reduce(matrix_shape, 1, true) == Shape{2, 1},
        "shape infers a kept reduction dimension");
    expect(
        reduce(matrix_shape, 0, false) == Shape{3},
        "shape infers a removed reduction dimension");
    expect_throws<std::invalid_argument>(
        [&]
        { require_reshape_compatible(matrix_shape, Shape{4, 2}); },
        "shape rejects reshape with a different element count");
    expect_throws<std::invalid_argument>(
        []
        { static_cast<void>(numel(Shape{2, -1})); },
        "shape geometry rejects negative extents");
    expect_throws<std::overflow_error>(
        []
        { static_cast<void>(numel(Shape{std::numeric_limits<minitensor::Index>::max(), 2})); },
        "shape geometry rejects overflowing element counts");

    const auto contiguous = Layout::contiguous(Shape{2, 3}, 3);
    expect(
        coordinates_from_linear(contiguous.shape(), 4) == Coordinates{1, 1},
        "linear ordinal converts to row-major coordinates");
    expect(
        contiguous.offset_from_coordinates(Coordinates{1, 1}) == 7,
        "coordinates include strides and base offset");

    const auto transformed_transpose = contiguous.transposed(0, 1);
    expect(
        transformed_transpose.shape() == Shape{3, 2} &&
            transformed_transpose.strides() == Strides{1, 3} &&
            transformed_transpose.offset() == 3,
        "layout owns transpose metadata transformation");

    const auto transformed_slice = contiguous.sliced(1, 0, 3, 2);
    expect(
        transformed_slice.shape() == Shape{2, 2} &&
            transformed_slice.strides() == Strides{3, 2} &&
            transformed_slice.offset() == 3,
        "layout owns slice metadata transformation");

    const auto transformed_reshape = contiguous.reshaped(Shape{3, 2});
    expect(
        transformed_reshape.shape() == Shape{3, 2} &&
            transformed_reshape.strides() == Strides{2, 1} &&
            transformed_reshape.offset() == 3,
        "layout owns contiguous reshape metadata transformation");

    const auto broadcast = Layout::contiguous(Shape{3}, 2).broadcast_to(Shape{2, 3});
    expect(
        broadcast.shape() == Shape{2, 3} &&
            broadcast.strides() == Strides{0, 1} &&
            broadcast.offset() == 2,
        "layout owns broadcast stride transformation");
    expect_throws<std::invalid_argument>(
        [&]
        { static_cast<void>(contiguous.broadcast_to(Shape{2, 2})); },
        "layout rejects an incompatible broadcast target");
    expect_throws<std::logic_error>(
        [&]
        { static_cast<void>(transformed_transpose.reshaped(Shape{2, 3})); },
        "layout rejects view-only reshape of non-contiguous metadata");

    const Layout transposed(Shape{3, 2}, Strides{1, 3}, 2);
    expect(
        coordinates_from_linear(transposed.shape(), 3) == Coordinates{1, 1},
        "linear conversion depends on shape, not physical strides");
    expect(
        transposed.offset_from_coordinates(Coordinates{1, 1}) == 6,
        "offset conversion honors non-contiguous strides");

    const auto scalar = Layout::contiguous(Shape{}, 5);
    expect(coordinates_from_linear(scalar.shape(), 0).empty(), "scalar coordinate is rank zero");
    expect(scalar.offset_from_coordinates(Coordinates{}) == 5, "scalar uses its base offset");

    expect_throws<std::out_of_range>(
        [&]
        { static_cast<void>(coordinates_from_linear(contiguous.shape(), -1)); },
        "negative linear ordinals are rejected");
    expect_throws<std::out_of_range>(
        [&]
        { static_cast<void>(coordinates_from_linear(contiguous.shape(), 6)); },
        "linear ordinals beyond numel are rejected");
    expect_throws<std::invalid_argument>(
        [&]
        { static_cast<void>(contiguous.offset_from_coordinates(Coordinates{1})); },
        "coordinate rank is validated");
    expect_throws<std::out_of_range>(
        [&]
        { static_cast<void>(contiguous.offset_from_coordinates(Coordinates{-1, 0})); },
        "negative coordinates are rejected");
    expect_throws<std::out_of_range>(
        [&]
        { static_cast<void>(contiguous.offset_from_coordinates(Coordinates{2, 0})); },
        "coordinates beyond an extent are rejected");
    expect_throws<std::out_of_range>(
        []
        {
            const auto empty = Layout::contiguous(Shape{0, 3});
            static_cast<void>(coordinates_from_linear(empty.shape(), 0));
        },
        "empty layouts have no valid linear ordinal");

    if (failures != 0)
    {
        std::cerr << failures << " layout test(s) failed\n";
        return 1;
    }
    std::cout << "All layout tests passed\n";
    return 0;
}
