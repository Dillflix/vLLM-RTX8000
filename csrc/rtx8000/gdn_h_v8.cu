// SPDX-License-Identifier: Apache-2.0
#include "gdn_h_kernel.cuh"

int gdn_h75_v8_launch(const void* k, const void* w, const void* u,
                      const float* g, const float* h0, const void* cu_seqlens,
                      const void* chunk_offsets, void* h, void* v_new,
                      float* ht, GdnH75Shape shape, cudaStream_t stream) {
  return gdn_h75::launch<8>(k, w, u, g, h0, cu_seqlens, chunk_offsets, h, v_new,
                            ht, shape, stream);
}
