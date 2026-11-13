# SPDX-License-Identifier: Apache-2.0
# The dispatch table is the canonical ordered list of prefill variants.
set(GLM_PACKED_BINDINGS "${CMAKE_CURRENT_LIST_DIR}/torch_bindings.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${GLM_PACKED_BINDINGS}")
file(READ "${GLM_PACKED_BINDINGS}" GLM_PACKED_BINDINGS_TEXT)
string(REGEX MATCH "prefill = \\{([^}]*)\\}" GLM_PACKED_TABLE
  "${GLM_PACKED_BINDINGS_TEXT}")
string(REGEX MATCHALL "v[0-9]+_m[0-9]+n[0-9]+_packed"
  GLM_PACKED_PREFILL_VARIANTS "${CMAKE_MATCH_1}")
if(NOT GLM_PACKED_PREFILL_VARIANTS)
  message(FATAL_ERROR "Could not read packed-dense prefill dispatch table")
endif()
set(GLM_PACKED_DENSE_SOURCES torch_bindings.cpp)
foreach(VARIANT IN LISTS GLM_PACKED_PREFILL_VARIANTS)
  string(REGEX REPLACE "^(v[0-9]+)_(.*)$" "\\1/\\2.cu"
    VARIANT_SOURCE "${VARIANT}")
  list(APPEND GLM_PACKED_DENSE_SOURCES "${VARIANT_SOURCE}")
endforeach()
list(APPEND GLM_PACKED_DENSE_SOURCES
  split/m16n64_packed.cu native/gemv_m1.cu)
list(TRANSFORM GLM_PACKED_DENSE_SOURCES
  PREPEND "${CMAKE_CURRENT_LIST_DIR}/")
add_library(glm_packed_dense SHARED ${GLM_PACKED_DENSE_SOURCES})
set_target_properties(glm_packed_dense PROPERTIES
  CUDA_ARCHITECTURES 75 CUDA_STANDARD 17 CXX_STANDARD 20
  CUDA_RUNTIME_LIBRARY Shared POSITION_INDEPENDENT_CODE ON)
if(DEFINED CUTLASS_INCLUDE_DIR)
  target_include_directories(glm_packed_dense PRIVATE "${CUTLASS_INCLUDE_DIR}")
else()
  target_include_directories(glm_packed_dense PRIVATE "${cutlass_SOURCE_DIR}/include")
endif()
target_include_directories(glm_packed_dense PRIVATE
  ${TORCH_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(glm_packed_dense PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
target_compile_options(glm_packed_dense PRIVATE -O3
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-static-global-template-stub=false>)
add_dependencies(_rtx8000_C glm_packed_dense)
install(TARGETS glm_packed_dense
  LIBRARY DESTINATION vllm/models/glm5next/nvidia/ops COMPONENT _rtx8000_C)
