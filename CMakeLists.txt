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
        src/tensor/primitives/manipulation/permute.cpp
        src/tensor/primitives/manipulation/reshape.cpp
        src/tensor/primitives/manipulation/slice.cpp
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
        src/nn/parameter.cpp
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
    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    
    # Declare function to add a test executable
    function(minitensor_add_test_executable target prefix label)
        # Add executable
        add_executable(${target} ${ARGN})
        # Link to library and set include directories
        target_link_libraries(${target} PRIVATE minitensor GTest::gtest_main)
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_compile_features(${target} PRIVATE cxx_std_20)
        set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)

        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        # Register with GoogleTest
        gtest_discover_tests(
            ${target}
            TEST_PREFIX "${prefix}."
            PROPERTIES LABELS "${label}"
        )
    endfunction()

    # Public tensor tests
    minitensor_add_test_executable(
        minitensor_tensor_public_tests
        public
        public
        tests/tensor/public/types_test.cpp
        tests/tensor/public/tensor_test.cpp
        tests/tensor/public/creation_test.cpp
        tests/tensor/public/elementwise_test.cpp
        tests/tensor/public/manipulation_test.cpp
        tests/tensor/public/reduction_test.cpp
        tests/tensor/public/linalg_test.cpp
        tests/tensor/public/data_test.cpp
        tests/tensor/public/evaluation_test.cpp
        tests/tensor/public/random_test.cpp
        tests/tensor/public/autograd_test.cpp
        tests/tensor/public/autograd_elementwise_test.cpp
        tests/tensor/public/autograd_manipulation_test.cpp
        tests/tensor/public/autograd_linalg_test.cpp
    )

    # Public neural-network tests
    minitensor_add_test_executable(
        minitensor_nn_public_tests
        nn
        public
        tests/nn/public/parameter_test.cpp
    )

    # Internal algorithm tests
    minitensor_add_test_executable(
        minitensor_tensor_internal_tests
        internal
        internal
        tests/tensor/internal/core/axis_test.cpp
        tests/tensor/internal/core/dense_size_test.cpp
        tests/tensor/internal/core/random_test.cpp
        tests/tensor/internal/core/shape_inference_test.cpp
        tests/tensor/internal/graph/graph_test.cpp
        tests/tensor/internal/graph/ownership_test.cpp
        tests/tensor/internal/storage/layout_test.cpp
        tests/tensor/internal/storage/materialization_test.cpp
        tests/tensor/internal/dispatch/kernel_registry_test.cpp
        tests/tensor/internal/dispatch/tensor_view_test.cpp
        tests/tensor/internal/execution/evaluator_test.cpp
        tests/tensor/internal/execution/runtime_registry_test.cpp
        tests/tensor/internal/autograd/engine_test.cpp
        tests/tensor/internal/autograd/reduce_to_shape_test.cpp
    )
    # Allow inclusion of internal source
    target_include_directories(
        minitensor_tensor_internal_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # CPU-specific backend tests
    minitensor_add_test_executable(
        minitensor_tensor_cpu_tests
        cpu
        cpu
        tests/tensor/cpu/runtime_test.cpp
        tests/tensor/cpu/registration_test.cpp
        tests/tensor/cpu/kernels_test.cpp
        tests/tensor/cpu/random_kernels_test.cpp
    )
    # Allow inclusion of internal source
    target_include_directories(
        minitensor_tensor_cpu_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
endif()
