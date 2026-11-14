cmake_minimum_required(VERSION 3.24...4.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(minitensor_cpp VERSION 0.1.0 LANGUAGES CXX)

# CUDA configuration
option(ENABLE_CUDA "Build the CUDA backend" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Sanitizer flag configuration
option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

if(ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "Sanitizers configured only for GCC")
    endif()

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>"
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all>"
    )

    add_link_options(
        "$<HOST_LINK:$<$<LINK_LANG_AND_ID:CXX,GNU>:-fsanitize=address;-fsanitize=undefined>>"
        "$<HOST_LINK:$<$<LINK_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-fsanitize=address;-Xcompiler=-fsanitize=undefined>>"
    )
endif()

# STATIC -> static library (linked at compile time)
add_library(minitensor STATIC)

if(ENABLE_CUDA)
    # Link the CUDA runtime.
    target_link_libraries(minitensor PRIVATE CUDA::cudart)

    # Preprocessor macro enabling CUDA for source files.
    target_compile_definitions(
        minitensor PRIVATE ENABLE_CUDA=1
    )

    # Set compilation standard.
    set_target_properties(minitensor PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RUNTIME_LIBRARY Shared
    )

    # Add CUDA-specific source.
    target_sources(minitensor PRIVATE
        src/tensor/backend/cuda/cuda_buffer.cpp
        src/tensor/backend/cuda/cuda_runtime.cpp
        src/tensor/backend/cuda/register_kernels.cpp
    )
endif()

# Specify all source files the library needs
target_sources(
    minitensor
    PRIVATE
        src/tensor/tensor.cpp
        src/tensor/types.cpp
        src/tensor/host_scalar.cpp
        src/tensor/data.cpp
        src/tensor/evaluation.cpp
        src/tensor/random.cpp
        src/tensor/core/axis.cpp
        src/tensor/core/reduction.cpp
        src/tensor/core/broadcasting.cpp
        src/tensor/apply_operation.cpp
        src/tensor/graph/apply_primitive.cpp
        src/tensor/graph/node.cpp
        src/tensor/graph/value.cpp
        src/tensor/graph/leaf.cpp
        src/tensor/ops/creation.cpp
        src/tensor/ops/elementwise.cpp
        src/tensor/ops/linalg.cpp
        src/tensor/ops/manipulation.cpp
        src/tensor/ops/reduction.cpp
        src/tensor/primitives/creation/full.cpp
        src/tensor/primitives/creation/uniform.cpp
        src/tensor/primitives/creation/normal.cpp
        src/tensor/primitives/elementwise/absolute_value.cpp
        src/tensor/primitives/elementwise/add.cpp
        src/tensor/primitives/elementwise/divide.cpp
        src/tensor/primitives/elementwise/logaddexp.cpp
        src/tensor/primitives/elementwise/exponential.cpp
        src/tensor/primitives/elementwise/hyperbolic_tangent.cpp
        src/tensor/primitives/elementwise/logarithm.cpp
        src/tensor/primitives/elementwise/multiply.cpp
        src/tensor/primitives/elementwise/negate.cpp
        src/tensor/primitives/elementwise/square_root.cpp
        src/tensor/primitives/elementwise/subtract.cpp
        src/tensor/primitives/elementwise/sign.cpp
        src/tensor/primitives/elementwise/sigmoid.cpp
        src/tensor/primitives/linalg/matmul.cpp
        src/tensor/primitives/manipulation/broadcast_to.cpp
        src/tensor/primitives/manipulation/concatenate.cpp
        src/tensor/primitives/manipulation/contiguous.cpp
        src/tensor/primitives/manipulation/permute.cpp
        src/tensor/primitives/manipulation/reshape.cpp
        src/tensor/primitives/manipulation/basic_index.cpp
        src/tensor/primitives/manipulation/gather.cpp
        src/tensor/primitives/reduction/sum.cpp
        src/tensor/primitives/reduction/logsumexp.cpp
        src/tensor/storage/layout.cpp
        src/tensor/storage/materialization.cpp
        src/tensor/storage/dense_size.cpp
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
        src/tensor/backend/cpu/kernels/linalg/kernels.cpp
        src/tensor/dispatch/kernel_registry.cpp
        src/tensor/dispatch/tensor_view.cpp
        src/tensor/execution/evaluator.cpp
        src/tensor/execution/runtime_registry.cpp
        src/tensor/execution/environment.cpp
        src/tensor/execution/host_transfer.cpp
        src/tensor/autograd/engine.cpp
        src/tensor/autograd/reduce_to_shape.cpp
        src/tensor/autograd/vjp_registry.cpp
        src/tensor/autograd/rules/elementwise.cpp
        src/tensor/autograd/rules/linalg.cpp
        src/tensor/autograd/rules/manipulation.cpp
        src/tensor/autograd/rules/reduction.cpp
        src/tensor/autograd.cpp
        src/nn/parameter.cpp
        src/nn/modules/linear.cpp
        src/nn/optimizers/adamw.cpp
        src/nn/optimizers/rmsprop.cpp
        src/nn/optimizers/sgd.cpp
        src/nn/loss.cpp
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

# Set compiler-specific flags
target_compile_options(minitensor PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/permissive->"
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
)

# Add demo executable
add_executable(minitensor_demo demo.cpp)
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
    )
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()
