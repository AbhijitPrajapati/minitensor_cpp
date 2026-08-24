#include "core/layout.hpp"

#include <functional>
#include <iostream>
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
    const auto contiguous = Layout::contiguous(Shape{2, 3}, 3);
    expect(
        contiguous.coordinates_from_linear(4) == Coordinates{1, 1},
        "linear ordinal converts to row-major coordinates");
    expect(
        contiguous.offset_from_coordinates(Coordinates{1, 1}) == 7,
        "coordinates include strides and base offset");

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
