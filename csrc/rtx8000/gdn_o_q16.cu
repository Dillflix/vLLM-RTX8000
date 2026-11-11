// SPDX-License-Identifier: Apache-2.0
#include "gdn_o_kernel.cuh"

int gdn_o75_q16_launch(const void* q, const void* k, const void* v,
                       const void* h, const float* g, const void* cu_seqlens,
                       const void* chunk_indices, void* out, Gdn75Shape shape,
                       cudaStream_t stream) {
  return gdn75::launch<16>(q, k, v, h, g, cu_seqlens, chunk_indices, out, shape,
                           stream);
}
