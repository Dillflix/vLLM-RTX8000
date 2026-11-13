// SPDX-License-Identifier: Apache-2.0
#include "specialized.cuh"
#include "iterator_decode.cuh"
extern "C" int v29_m128n256_packed_configure(int* stats) {
  return glm_lossless_cutlass_v29::Specialized<128, 256>::configure(stats);
}
extern "C" int v29_m128n256_packed(void const* x, void const* w,
                                   void const* meta, void* out, int m, int n,
                                   int k, void* stream) {
  return glm_lossless_cutlass_v29::launch_specialized<128, 256>(
      x, w, meta, out, m, n, k, stream);
}
extern "C" int v29_m128n256_packed_decode(void const* w, void const* meta,
                                          void* out, int n, int k, int clear,
                                          void* stream) {
  return glm_lossless_cutlass_v29::launch_decode<128, 256>(w, meta, out, n, k,
                                                           clear, stream);
}
