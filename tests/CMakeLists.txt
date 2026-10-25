# Function to add test executable
function(minitensor_add_test_executable target test_name labels)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE minitensor GTest::gtest_main)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    target_compile_features(${target} PRIVATE cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)

    # Same compiler-specific flags for MSVC
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

minitensor_add_test_executable(
    minitensor_tensor_public_tests
    tensor.public
    "tensor;public"
    tensor/public/types_test.cpp
    tensor/public/host_scalar_test.cpp
    tensor/public/tensor_test.cpp
    tensor/public/creation_test.cpp
    tensor/public/elementwise_test.cpp
    tensor/public/manipulation_test.cpp
    tensor/public/reduction_test.cpp
    tensor/public/linalg_test.cpp
    tensor/public/data_test.cpp
    tensor/public/evaluation_test.cpp
    tensor/public/random_test.cpp
    tensor/public/autograd_test.cpp
)

minitensor_add_test_executable(
    minitensor_tensor_conformance_cpu_tests
    tensor.conformance.cpu
    "tensor;conformance;cpu"
    tensor/conformance/autograd/elementwise_test.cpp
    tensor/conformance/autograd/linalg_test.cpp
    tensor/conformance/autograd/manipulation_test.cpp
    tensor/conformance/autograd/reduction_test.cpp
)
target_compile_definitions(
    minitensor_tensor_conformance_cpu_tests
    PRIVATE MINITENSOR_TEST_CONFORMANCE_CPU
)

minitensor_add_test_executable(
    minitensor_nn_public_tests
    nn.public
    "nn;public"
    nn/public/parameter_test.cpp
    nn/public/parameter_tree_test.cpp
    nn/public/gradients_test.cpp
    nn/public/leafification_test.cpp
    nn/public/linear_test.cpp
    nn/public/loss_test.cpp
    nn/public/adamw_test.cpp
    nn/public/sgd_test.cpp
)

minitensor_add_test_executable(
    minitensor_tensor_internal_tests
    tensor.internal
    "tensor;internal"
    tensor/internal/core/axis_test.cpp
    tensor/internal/core/broadcasting_test.cpp
    tensor/internal/core/reduction_test.cpp
    tensor/internal/graph/graph_test.cpp
    tensor/internal/graph/ownership_test.cpp
    tensor/internal/graph/primitive_id_test.cpp
    tensor/internal/storage/layout_test.cpp
    tensor/internal/storage/materialization_test.cpp
    tensor/internal/storage/dense_size_test.cpp
    tensor/internal/dispatch/kernel_registry_test.cpp
    tensor/internal/dispatch/tensor_view_test.cpp
    tensor/internal/execution/evaluator_test.cpp
    tensor/internal/execution/runtime_registry_test.cpp
    tensor/internal/autograd/engine_test.cpp
    tensor/internal/autograd/reduce_to_shape_test.cpp
    tensor/internal/autograd/vjp_registry_test.cpp
)
target_include_directories(
    minitensor_tensor_internal_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)

minitensor_add_test_executable(
    minitensor_tensor_cpu_tests
    tensor.backend.cpu
    "tensor;backend;cpu"
    tensor/backend/cpu/runtime_test.cpp
    tensor/backend/cpu/registration_test.cpp
    tensor/backend/cpu/iteration/elementwise_test.cpp
    tensor/backend/cpu/iteration/reduction_test.cpp
    tensor/backend/cpu/iteration/matmul_test.cpp
    tensor/backend/cpu/kernels/creation_test.cpp
    tensor/backend/cpu/kernels/elementwise_test.cpp
    tensor/backend/cpu/kernels/copy_test.cpp
    tensor/backend/cpu/kernels/reduction_test.cpp
    tensor/backend/cpu/kernels/linalg_test.cpp
)
target_include_directories(
    minitensor_tensor_cpu_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
