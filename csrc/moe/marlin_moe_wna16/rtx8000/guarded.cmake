# SPDX-License-Identifier: Apache-2.0
# Shared generation and source list for the full and incremental builds.
set(RTX8000_MARLIN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(RTX8000_MARLIN_GENERATED_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/rtx8000_marlin_generated")
set(RTX8000_MARLIN_GUARDED_TEMPLATE
  "${RTX8000_MARLIN_GENERATED_DIR}/guarded_marlin_template.cuh")
set(RTX8000_MARLIN_GUARDED_MANIFEST
  "${RTX8000_MARLIN_GENERATED_DIR}/source-manifest.json")
add_custom_command(OUTPUT "${RTX8000_MARLIN_GUARDED_TEMPLATE}"
  BYPRODUCTS "${RTX8000_MARLIN_GUARDED_MANIFEST}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${RTX8000_MARLIN_GENERATED_DIR}"
  COMMAND ${RTX8000_MARLIN_PYTHON}
    "${RTX8000_MARLIN_SOURCE_DIR}/safety/generate_template.py"
    --source "${RTX8000_MARLIN_SOURCE_DIR}/marlin_template.h"
    --output "${RTX8000_MARLIN_GUARDED_TEMPLATE}"
    --manifest "${RTX8000_MARLIN_GUARDED_MANIFEST}"
  DEPENDS "${RTX8000_MARLIN_SOURCE_DIR}/safety/generate_template.py"
    "${RTX8000_MARLIN_SOURCE_DIR}/marlin_template.h"
  VERBATIM)
set(RTX8000_MARLIN_CUDA_SOURCES
  "${RTX8000_MARLIN_SOURCE_DIR}/rtx8000_dispatch.cu"
  "${RTX8000_MARLIN_SOURCE_DIR}/rtx8000_guarded_selector.cu"
  "${RTX8000_MARLIN_SOURCE_DIR}/rtx8000_guarded_m16.cu"
  "${RTX8000_MARLIN_SOURCE_DIR}/rtx8000_guarded_m32.cu")
