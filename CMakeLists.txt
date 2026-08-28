cmake_minimum_required(VERSION 4.0.2)

project(minitensor_cpp VERSION 0.1.0 LANGUAGES CXX)

# STATIC -> static library (linked at compile time)
add_library(minitensor STATIC)

# Specify all source files the library needs
target_sources(
    minitensor
    PRIVATE
        src/tensor/tensor.cpp
        src/tensor/types.cpp
        src/tensor/core/broadcast_shape.cpp
        src/tensor/graph/apply_operation.cpp
        src/tensor/graph/ids.cpp
        src/tensor/graph/node.cpp
        src/tensor/graph/value.cpp
        src/tensor/ops/add.cpp
        src/tensor/ops/full.cpp
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
endif()
