#include <exception>
#include <iostream>
#include <string_view>

#include "support/test.hpp"

namespace minitensor::test
{
    void run_types_test();
    void run_operations_test();
    void run_layout_test();
    void run_materialization_test();
    void run_graph_objects_test();
    void run_apply_operation_test();
    void run_ownership_test();
    void run_cpu_runtime_test();
}

namespace
{
    struct TestEntry final
    {
        std::string_view name;
        void (*run)();
    };
}

int main()
{
    using namespace minitensor::test;

    const TestEntry tests[]{
        {"public types", run_types_test},
        {"public operations and tensor handles", run_operations_test},
        {"layout", run_layout_test},
        {"materialization", run_materialization_test},
        {"graph objects", run_graph_objects_test},
        {"apply operation", run_apply_operation_test},
        {"ownership", run_ownership_test},
        {"cpu runtime", run_cpu_runtime_test},
    };

    int failures = 0;
    for (const TestEntry &test : tests)
    {
        try
        {
            test.run();
            std::cout << "PASS: " << test.name << '\n';
        }
        catch (const TestFailure &error)
        {
            ++failures;
            std::cerr << "FAIL: " << test.name << ": " << error.what() << '\n';
        }
        catch (const std::exception &error)
        {
            ++failures;
            std::cerr << "FAIL: " << test.name << ": unexpected exception: " << error.what() << '\n';
        }
        catch (...)
        {
            ++failures;
            std::cerr << "FAIL: " << test.name << ": unexpected non-standard exception\n";
        }
    }

    if (failures != 0)
    {
        std::cerr << failures << " test group(s) failed\n";
        return 1;
    }

    std::cout << "All minitensor tests passed\n";
    return 0;
}
