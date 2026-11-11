// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

struct Gdn75Shape {
  int64_t tokens;
  int chunks;
  int sequences;
  bool indices64;
  float scale;
};

int gdn_o75_q16_launch(const void* q, const void* k, const void* v,
                       const void* h, const float* g, const void* cu_seqlens,
                       const void* chunk_indices, void* out, Gdn75Shape shape,
                       cudaStream_t stream);
int gdn_o75_q32_launch(const void* q, const void* k, const void* v,
                       const void* h, const float* g, const void* cu_seqlens,
                       const void* chunk_indices, void* out, Gdn75Shape shape,
                       cudaStream_t stream);
