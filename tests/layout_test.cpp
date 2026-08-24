#include "core/layout.hpp"

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
    expect(matrix_shape.rank() == 2, "shape owns its rank");
    expect(matrix_shape.numel() == 6, "shape owns its checked element count");
    expect(Shape{}.numel() == 1, "rank-zero shape represents one scalar element");
    expect(
        Shape{2, 1, 3}.broadcast_with(Shape{4, 3}) == Shape{2, 4, 3},
        "shape infers a trailing-dimension broadcast result");
    expect(
        Shape{3}.is_broadcastable_to(Shape{2, 3}),
        "shape recognizes a valid broadcast target");
    expect(
        matrix_shape.reduced(1, true) == Shape{2, 1},
        "shape infers a kept reduction dimension");
    expect(
        matrix_shape.reduced(0, false) == Shape{3},
        "shape infers a removed reduction dimension");
    expect(
        matrix_shape.is_reshape_compatible_with(Shape{3, 2}),
        "shape recognizes reshape compatibility");
    expect_throws<std::invalid_argument>(
        []
        { static_cast<void>(Shape{2, -1}); },
        "shape rejects negative extents at construction");
    expect_throws<std::overflow_error>(
        []
        { static_cast<void>(Shape{std::numeric_limits<minitensor::Index>::max(), 2}); },
        "shape rejects overflowing element counts at construction");

    const auto contiguous = Layout::contiguous(Shape{2, 3}, 3);
    expect(
        contiguous.coordinates_from_linear(4) == Coordinates{1, 1},
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
        transposed.coordinates_from_linear(3) == Coordinates{1, 1},
        "linear conversion depends on shape, not physical strides");
    expect(
        transposed.offset_from_coordinates(Coordinates{1, 1}) == 6,
        "offset conversion honors non-contiguous strides");

    const auto scalar = Layout::contiguous(Shape{}, 5);
    expect(scalar.coordinates_from_linear(0).empty(), "scalar coordinate is rank zero");
    expect(scalar.offset_from_coordinates(Coordinates{}) == 5, "scalar uses its base offset");

    expect_throws<std::out_of_range>(
        [&]
        { static_cast<void>(contiguous.coordinates_from_linear(-1)); },
        "negative linear ordinals are rejected");
    expect_throws<std::out_of_range>(
        [&]
        { static_cast<void>(contiguous.coordinates_from_linear(6)); },
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
            static_cast<void>(empty.coordinates_from_linear(0));
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
