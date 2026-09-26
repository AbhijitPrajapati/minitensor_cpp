#pragma once

#include <cmath>
#include <span>
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
			throw TestFailure{ std::move(message) };
		}
	}

	inline void expect_near(
		float actual,
		float expected,
		float tolerance,
		std::string message)
	{
		if (actual != expected &&
			(!std::isfinite(actual) ||
			!std::isfinite(expected) ||
			std::fabs(actual - expected) > tolerance))
		{
			throw TestFailure{ std::move(message) };
		}
	}

	inline void expect_near(
		std::span<const float> actual,
		std::span<const float> expected,
		float tolerance,
		std::string message)
	{
		if (actual.size() != expected.size())
		{
			throw TestFailure{ std::move(message) };
		}

		for (std::size_t index = 0; index < actual.size(); ++index)
		{
			if (actual[index] != expected[index] &&
				(!std::isfinite(actual[index]) ||
				!std::isfinite(expected[index]) ||
				std::fabs(actual[index] - expected[index]) > tolerance))
			{
				throw TestFailure{ std::move(message) };
			}
		}
	}

	template <typename Exception, typename Action>
	void expect_throws(Action&& action, const std::string& message)
	{
		try
		{
			std::forward<Action>(action)();
		}
		catch (const Exception&)
		{
			return;
		}
		catch (...)
		{
			throw TestFailure{ message + " (unexpected exception type)" };
		}

		throw TestFailure{ message + " (no exception thrown)" };
	}
}
