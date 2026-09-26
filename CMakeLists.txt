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
        src/tensor/data.cpp
        src/tensor/evaluation.cpp
        src/tensor/random.cpp
        src/tensor/core/shape_inference.cpp
        src/tensor/core/dense_size.cpp
        src/tensor/core/axis.cpp
        src/tensor/core/random.cpp
        src/tensor/graph/apply_operation.cpp
        src/tensor/graph/node.cpp
        src/tensor/graph/value.cpp
        src/tensor/ops/creation.cpp
        src/tensor/ops/elementwise.cpp
        src/tensor/ops/linalg.cpp
        src/tensor/ops/manipulation.cpp
        src/tensor/ops/reduction.cpp
        src/tensor/primitives/creation/full.cpp
        src/tensor/primitives/creation/uniform.cpp
        src/tensor/primitives/creation/normal.cpp
        src/tensor/primitives/elementwise/add.cpp
        src/tensor/primitives/elementwise/divide.cpp
        src/tensor/primitives/elementwise/exponential.cpp
        src/tensor/primitives/elementwise/hyperbolic_tangent.cpp
        src/tensor/primitives/elementwise/logarithm.cpp
        src/tensor/primitives/elementwise/multiply.cpp
        src/tensor/primitives/elementwise/negate.cpp
        src/tensor/primitives/elementwise/square_root.cpp
        src/tensor/primitives/elementwise/subtract.cpp
        src/tensor/primitives/linalg/matmul.cpp
        src/tensor/primitives/manipulation/broadcast_to.cpp
        src/tensor/primitives/manipulation/concatenate.cpp
        src/tensor/primitives/manipulation/contiguous.cpp
        src/tensor/primitives/manipulation/narrow.cpp
        src/tensor/primitives/manipulation/permute.cpp
        src/tensor/primitives/manipulation/reshape.cpp
        src/tensor/primitives/reduction/sum.cpp
        src/tensor/storage/layout.cpp
        src/tensor/storage/materialization.cpp
        src/tensor/backend/cpu/cpu_buffer.cpp
        src/tensor/backend/cpu/cpu_runtime.cpp
        src/tensor/backend/cpu/register_kernels.cpp
        src/tensor/backend/cpu/detail/random_engine.cpp
        src/tensor/backend/cpu/iteration/elementwise.cpp
        src/tensor/backend/cpu/iteration/matmul.cpp
        src/tensor/backend/cpu/iteration/reduction.cpp
        src/tensor/backend/cpu/kernels/creation/kernels.cpp
        src/tensor/backend/cpu/kernels/elementwise/kernels.cpp
        src/tensor/backend/cpu/kernels/copy/kernels.cpp
        src/tensor/backend/cpu/kernels/reduction/kernels.cpp
        src/tensor/backend/cpu/kernels/linalg/matmul.cpp
        src/tensor/dispatch/kernel_registry.cpp
        src/tensor/dispatch/tensor_view.cpp
        src/tensor/execution/evaluator.cpp
        src/tensor/execution/generator_registry.cpp
        src/tensor/execution/runtime_registry.cpp
        src/tensor/execution/environment.cpp
        src/tensor/autograd/engine.cpp
        src/tensor/autograd/reduce_to_shape.cpp
        src/tensor/autograd.cpp
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
    add_executable(
    tensor_tests 
    tests/tensor/test_main.cpp
    tests/tensor/public/operations_test.cpp
    tests/tensor/public/types_test.cpp
    tests/tensor/public/evaluation_test.cpp
    tests/tensor/public/data_test.cpp
    tests/tensor/public/autograd_test.cpp
    tests/tensor/internal/core_test.cpp
    tests/tensor/internal/apply_operation_test.cpp
    tests/tensor/internal/graph_objects_test.cpp
    tests/tensor/internal/layout_test.cpp
    tests/tensor/internal/materialization_test.cpp
    tests/tensor/internal/ownership_test.cpp
    tests/tensor/internal/cpu_runtime_test.cpp
    tests/tensor/internal/cpu_random_test.cpp
    tests/tensor/internal/kernel_registry_test.cpp
    tests/tensor/internal/tensor_view_test.cpp
    tests/tensor/internal/cpu_kernels_test.cpp
    tests/tensor/internal/matmul_plan_test.cpp
    tests/tensor/internal/evaluator_test.cpp
    tests/tensor/internal/runtime_registry_test.cpp
    tests/tensor/internal/random_test.cpp
    tests/tensor/internal/reduce_to_shape_test.cpp
    tests/tensor/internal/autograd_engine_test.cpp
    )
    # Link test executable to library
    target_link_libraries(tensor_tests PRIVATE minitensor)
    # Allow the tests to access internal headers
    target_include_directories(tensor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # Add test w/ command to run
    add_test(NAME minitensor.tensor COMMAND tensor_tests)
endif()
