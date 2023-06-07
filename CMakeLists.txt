cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cudnn-perf-tests CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
	COMMAND "${Python3_EXECUTABLE}" -m venv "${CMAKE_CURRENT_BINARY_DIR}/venv"
	COMMAND_ECHO STDOUT
	COMMAND_ERROR_IS_FATAL ANY)
execute_process(
	COMMAND "${CMAKE_CURRENT_BINARY_DIR}/venv/bin/python" -m pip install torch torchvision torchaudio
	COMMAND_ECHO STDOUT
	COMMAND_ERROR_IS_FATAL ANY)

file(GLOB_RECURSE CUDNN_INCLUDE "cudnn.h")
get_filename_component(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE} DIRECTORY)
file(GLOB_RECURSE CUDNN_LIBRARIES "libcudnn.so*")

add_executable(${PROJECT_NAME} src/benchmark.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDNN_INCLUDE_DIRS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
target_link_libraries(${PROJECT_NAME} CUDA::curand ${CUDNN_LIBRARIES})

include(CTest)

add_test(NAME ${PROJECT_NAME}_fp32 COMMAND $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/conv_example.txt fp32.csv fp32 0 100 NHWC NHWC NHWC)
add_test(NAME ${PROJECT_NAME}_fp16 COMMAND $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/conv_example.txt fp16.csv fp16 0 100 NHWC NHWC NHWC)
add_test(NAME ${PROJECT_NAME}_int8 COMMAND $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/conv_example.txt int8.csv int8 0 100 NHWC NHWC NHWC)
add_test(NAME ${PROJECT_NAME}_int8x4 COMMAND $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/conv_example.txt int8x4.csv int8x4 0 100 NCHW_VECT_C NCHW_VECT_C NCHW_VECT_C)
add_test(NAME ${PROJECT_NAME}_int8x32 COMMAND $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/conv_example.txt int8x32.csv int8x32 0 100 NCHW_VECT_C NCHW_VECT_C NCHW_VECT_C)

