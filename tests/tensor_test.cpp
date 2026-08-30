#include <minitensor/minitensor.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

// This is an old test file that has nothing to do with the code that is to be tested.
// You do not need to worry about this file. Just ignore it.

namespace
{

    int failures = 0;

    void fail(const std::string &message)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }

    void expect(const bool condition, const std::string &message)
    {
        if (!condition)
        {
            fail(message);
        }
    }

    void expect_close(
        const float actual,
        const float expected,
        const std::string &message,
        const float tolerance = 1.0e-5F)
    {
        if (std::fabs(actual - expected) > tolerance)
        {
            fail(message + " (expected " + std::to_string(expected) +
                 ", got " + std::to_string(actual) + ")");
        }
    }

    template <typename Exception>
    void expect_throws(const std::function<void()> &action, const std::string &message)
    {
        try
        {
            action();
            fail(message + " (no exception was thrown)");
        }
        catch (const Exception &)
        {
        }
        catch (...)
        {
            fail(message + " (wrong exception type)");
        }
    }

} // namespace

int main()
{

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All minitensor tests passed\n";
    return 0;
}
