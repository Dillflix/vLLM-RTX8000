// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

struct GdnH75Shape {
  int64_t tokens;
  int64_t chunks;
  int sequences;
  bool sequences64;
  bool chunks64;
};

using GdnH75Launch = int (*)(const void*, const void*, const void*,
                             const float*, const float*, const void*,
                             const void*, void*, void*, float*, GdnH75Shape,
                             cudaStream_t);

int gdn_h75_v8_launch(const void* k, const void* w, const void* u,
                      const float* g, const float* h0, const void* cu_seqlens,
                      const void* chunk_offsets, void* h, void* v_new,
                      float* ht, GdnH75Shape shape, cudaStream_t stream);
int gdn_h75_v16_launch(const void* k, const void* w, const void* u,
                       const float* g, const float* h0, const void* cu_seqlens,
                       const void* chunk_offsets, void* h, void* v_new,
                       float* ht, GdnH75Shape shape, cudaStream_t stream);
