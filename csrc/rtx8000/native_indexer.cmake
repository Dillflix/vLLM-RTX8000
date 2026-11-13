# SPDX-License-Identifier: Apache-2.0
# The ctypes adapter and its exact-arithmetic reference share these C entry points.
add_library(glm_indexer SHARED
  "${CMAKE_CURRENT_LIST_DIR}/native_4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/native_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/native_fp8decoded.cu")
set_target_properties(glm_indexer PROPERTIES
  CUDA_ARCHITECTURES 75 CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_compile_options(glm_indexer PRIVATE -O3
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_dependencies(_rtx8000_C glm_indexer)
install(TARGETS glm_indexer
  LIBRARY DESTINATION vllm/models/glm5next/nvidia/ops COMPONENT _rtx8000_C)
