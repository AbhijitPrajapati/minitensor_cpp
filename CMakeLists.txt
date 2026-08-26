cmake_minimum_required(VERSION 4.0.2)

project(minitensor_cpp VERSION 0.1.0 LANGUAGES CXX)

# STATIC -> static library (linked at compile time)
add_library(minitensor STATIC)

# Specify all source files the library needs
target_sources(
    minitensor
    PRIVATE
        src/tensor.cpp
        src/autograd/engine.cpp
        src/autograd/recording.cpp
        src/autograd/tensor_access.cpp
        src/core/layout.cpp
        src/core/shape.cpp
        src/core/storage.cpp
        src/core/tensor_impl.cpp
        src/kernels/iteration.cpp
        src/kernels/kernels.cpp
        src/kernels/tensor_view.cpp
        src/ops/activations.cpp
        src/ops/elementwise.cpp
        src/ops/matmul.cpp
        src/ops/reduction.cpp
        src/ops/views.cpp
)

# Specify all header files the library needs
target_include_directories(
    minitensor
    PUBLIC
        # Any target that links the library will also require the public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        # Only the library requires the private headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Specify C++ 20 support
target_compile_features(minitensor PUBLIC cxx_std_20)
# No compiler-specific extentions are used
set_target_properties(minitensor PROPERTIES CXX_EXTENSIONS OFF)

# Set compile-specific flags to catch mistakes
if(MSVC)
    target_compile_options(minitensor PRIVATE /W4 /permissive-)
else()
    target_compile_options(minitensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add demo executable
add_executable(minitensor_demo main.cpp)
# Link demo executable to library
target_link_libraries(minitensor_demo PRIVATE minitensor)

# Enable testing with CTest
include(CTest)
if(BUILD_TESTING)
    # Add test executable
    add_executable(minitensor_tests tests/test_main.cpp)
    # Link test executable to library
    target_link_libraries(minitensor_tests PRIVATE minitensor)
    # Add test w/ command to run
    add_test(NAME minitensor_tests COMMAND minitensor_tests)

    # Internal geometry tests exercise shape validation and Layout transformations.
    add_executable(minitensor_layout_tests tests/layout_test.cpp)
    target_link_libraries(minitensor_layout_tests PRIVATE minitensor)
    target_include_directories(
        minitensor_layout_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME minitensor_layout_tests COMMAND minitensor_layout_tests)

    # Kernel iteration tests exercise elementwise, reduction, and contraction traversal.
    add_executable(minitensor_iteration_tests tests/iteration_test.cpp)
    target_link_libraries(minitensor_iteration_tests PRIVATE minitensor)
    target_include_directories(
        minitensor_iteration_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME minitensor_iteration_tests COMMAND minitensor_iteration_tests)

    # Autograd contract tests verify recording, explicit saved state, and no-grad execution.
    add_executable(minitensor_autograd_tests tests/autograd_test.cpp)
    target_link_libraries(minitensor_autograd_tests PRIVATE minitensor)
    target_include_directories(
        minitensor_autograd_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME minitensor_autograd_tests COMMAND minitensor_autograd_tests)
endif()
