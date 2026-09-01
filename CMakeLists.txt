cmake_minimum_required(VERSION 3.16)
project(picotorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(picotorch STATIC
    src/tensor.cpp
    src/module.cpp
    src/ops/linear.cpp
    src/ops/layernorm.cpp
    src/ops/activation.cpp
    src/kernels/mha.cpp
    src/kernels/softmax.cpp
    src/encoder.cpp
    src/sequential.cpp
    src/backends/ref.cpp
    src/backends/s3_fp32.cpp
)
target_include_directories(picotorch PUBLIC include)

function(picotorch_example name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE picotorch)
    target_include_directories(${name} PRIVATE include)
endfunction()

picotorch_example(01_linear examples/minimal/01_linear.cpp)
picotorch_example(02_sequential examples/minimal/02_sequential.cpp)
picotorch_example(03_self_attention examples/minimal/03_self_attention.cpp)
picotorch_example(04_cross_attention examples/minimal/04_cross_attention.cpp)
picotorch_example(05_encoder_layer examples/minimal/05_encoder_layer.cpp)

add_executable(test_mha_maxabs host/test_mha_maxabs.cpp)
target_link_libraries(test_mha_maxabs PRIVATE picotorch)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/wisdm/wisdm_weights.hpp")
    picotorch_example(wisdm_infer examples/wisdm/wisdm_infer.cpp)
    target_include_directories(wisdm_infer PRIVATE examples/wisdm)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cifar10/cifar_weights.hpp")
    picotorch_example(cifar_infer examples/cifar10/cifar_infer.cpp)
    target_include_directories(cifar_infer PRIVATE examples/cifar10)
endif()

set(RICE_INCLUDE "" CACHE PATH "Path to rice-s3/firmware/include for host golden")
if(RICE_INCLUDE)
    get_filename_component(RICE_FW "${RICE_INCLUDE}" DIRECTORY)
    add_executable(host_golden
        examples/pico-former/host_golden.cpp
        ${RICE_FW}/src/ceqt_infer.cpp
    )
    target_link_libraries(host_golden PRIVATE picotorch)
    target_include_directories(host_golden PRIVATE ${RICE_INCLUDE} include)
    target_compile_definitions(host_golden PRIVATE PICOTORCH_HOST_GOLDEN=1)
endif()
