# SPDX-License-Identifier: Apache-2.0
# Standalone incremental build for an existing editable SM75 installation.
cmake_minimum_required(VERSION 3.26)
project(vllm_rtx8000 LANGUAGES CXX CUDA)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
# This standalone project builds only SM75; prevent Torch from inferring
# architectures from the build host (or using its broad no-device fallback).
set(ENV{TORCH_CUDA_ARCH_LIST} "7.5")
find_package(Torch REQUIRED)
if(NOT DEFINED CUTLASS_INCLUDE_DIR)
  message(FATAL_ERROR "Set CUTLASS_INCLUDE_DIR to the existing CUTLASS include directory")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/qsa_variants.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dense_mma_variants.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dense_gemv_extra_variants.cmake")
add_library(_rtx8000_C MODULE torch_bindings.cpp qsa.cu ${RTX8000_QSA_SOURCES}
  dense.cu dense_mma.cu ${RTX8000_DENSE_MMA_SOURCES}
  ${RTX8000_DENSE_GEMV_EXTRA_SOURCES}
  lossless_head_bindings.cpp lossless_head_tile.cu
  gdn_o_bindings.cpp gdn_o_q16.cu gdn_o_q32.cu
  gdn_h_bindings.cpp gdn_h_v8.cu gdn_h_v16.cu)
target_include_directories(_rtx8000_C PRIVATE ${TORCH_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR})
target_link_libraries(_rtx8000_C PRIVATE ${TORCH_LIBRARIES} Python3::Module)
set_target_properties(_rtx8000_C PROPERTIES PREFIX "" SUFFIX ".abi3.so"
  CUDA_ARCHITECTURES 75 CUDA_STANDARD 17 CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)
target_compile_options(_rtx8000_C PRIVATE -O3 $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(_rtx8000_C PRIVATE Py_LIMITED_API=0x03090000)
install(TARGETS _rtx8000_C LIBRARY DESTINATION vllm COMPONENT _rtx8000_C)
include("${CMAKE_CURRENT_LIST_DIR}/native_indexer.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/packed_dense/packed_dense.cmake")
