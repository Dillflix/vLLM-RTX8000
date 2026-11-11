# SPDX-License-Identifier: Apache-2.0
# Isolated experiment: no install rule and no production target changes.
cmake_minimum_required(VERSION 3.26)
project(rtx8000_marlin_safety LANGUAGES CXX CUDA)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)
get_filename_component(VLLM_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(GENERATED_TEMPLATE "${CMAKE_CURRENT_BINARY_DIR}/guarded_marlin_template.cuh")
add_custom_command(OUTPUT "${GENERATED_TEMPLATE}"
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_LIST_DIR}/generate_template.py"
    --source "${CMAKE_CURRENT_LIST_DIR}/../marlin_template.h"
    --output "${GENERATED_TEMPLATE}"
    --manifest "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.json"
  DEPENDS generate_template.py ../marlin_template.h
  VERBATIM)
add_library(_rtx8000_marlin_safety_C MODULE
  bindings.cpp control.cu guarded.cu "${GENERATED_TEMPLATE}")
target_include_directories(_rtx8000_marlin_safety_C PRIVATE
  "${VLLM_SOURCE_ROOT}/csrc" "${CMAKE_CURRENT_BINARY_DIR}" ${TORCH_INCLUDE_DIRS})
target_link_libraries(_rtx8000_marlin_safety_C PRIVATE ${TORCH_LIBRARIES})
target_compile_options(_rtx8000_marlin_safety_C PRIVATE
  -O3 "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v>")
set_target_properties(_rtx8000_marlin_safety_C PROPERTIES
  PREFIX "" SUFFIX ".so" CUDA_ARCHITECTURES 75
  CUDA_STANDARD 17 CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)
