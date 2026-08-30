#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace minitensor::test
{
    class TestFailure final : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    inline void expect(bool condition, std::string message)
    {
        if (!condition)
        {
            throw TestFailure{std::move(message)};
        }
    }

    template <typename Exception, typename Action>
    void expect_throws(Action &&action, const std::string &message)
    {
        try
        {
            std::forward<Action>(action)();
        }
        catch (const Exception &)
        {
            return;
        }
        catch (...)
        {
            throw TestFailure{message + " (unexpected exception type)"};
        }

        throw TestFailure{message + " (no exception thrown)"};
    }
}
