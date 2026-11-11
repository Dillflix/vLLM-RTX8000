# SPDX-License-Identifier: Apache-2.0
# Incremental companion build for an existing, compatible SM75 installation.
# The full repository build defines the same target in its top-level CMake.
cmake_minimum_required(VERSION 3.26)
project(vllm_rtx8000_marlin LANGUAGES CXX CUDA)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)

if(NOT DEFINED VLLM_MOE_LIBRARY OR NOT EXISTS "${VLLM_MOE_LIBRARY}")
  message(FATAL_ERROR "Set VLLM_MOE_LIBRARY to the installed _moe_C_stable_libtorch extension")
endif()
get_filename_component(MOE_LIBRARY_DIR "${VLLM_MOE_LIBRARY}" DIRECTORY)
get_filename_component(MOE_LIBRARY_NAME "${VLLM_MOE_LIBRARY}" NAME)
if(NOT MOE_LIBRARY_NAME STREQUAL "_moe_C_stable_libtorch.abi3.so")
  message(FATAL_ERROR "VLLM_MOE_LIBRARY must name _moe_C_stable_libtorch.abi3.so")
endif()
get_filename_component(VLLM_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

set(RTX8000_MARLIN_PYTHON "${Python3_EXECUTABLE}")
include("${CMAKE_CURRENT_LIST_DIR}/guarded.cmake")
add_library(_rtx8000_marlin_C MODULE ${RTX8000_MARLIN_CUDA_SOURCES}
  "${RTX8000_MARLIN_GUARDED_TEMPLATE}")
target_include_directories(_rtx8000_marlin_C PRIVATE
  "${VLLM_SOURCE_ROOT}/csrc" "${RTX8000_MARLIN_GENERATED_DIR}"
  ${TORCH_INCLUDE_DIRS})
# Link by basename: DT_NEEDED must remain relocatable beside the original
# extension, rather than recording an absolute build-machine path.
target_link_directories(_rtx8000_marlin_C PRIVATE "${MOE_LIBRARY_DIR}")
target_link_libraries(_rtx8000_marlin_C PRIVATE
  ${TORCH_LIBRARIES} Python3::Module
  "-Wl,--no-as-needed" "-l:${MOE_LIBRARY_NAME}" "-Wl,--as-needed")
target_compile_definitions(_rtx8000_marlin_C PRIVATE
  USE_CUDA TORCH_TARGET_VERSION=0x020B000000000000ULL
  Py_LIMITED_API=0x03090000 TORCH_EXTENSION_NAME=_rtx8000_marlin_C)
target_compile_options(_rtx8000_marlin_C PRIVATE
  -O3 "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-v>")
set_target_properties(_rtx8000_marlin_C PROPERTIES
  PREFIX "" SUFFIX ".abi3.so" CUDA_ARCHITECTURES 75
  CUDA_STANDARD 20 CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${MOE_LIBRARY_DIR}" INSTALL_RPATH "$ORIGIN")
install(TARGETS _rtx8000_marlin_C LIBRARY DESTINATION vllm
  COMPONENT _rtx8000_marlin_C)
