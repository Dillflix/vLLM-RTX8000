// SPDX-License-Identifier: Apache-2.0
#include "specialized.cuh"
#include "iterator_decode.cuh"
extern "C" int v30_m32n128_packed_configure(int* stats) {
  return glm_lossless_cutlass_v30::Specialized<32, 128>::configure(stats);
}
extern "C" int v30_m32n128_packed(void const* x, void const* w,
                                  void const* meta, void* out, int m, int n,
                                  int k, void* stream) {
  return glm_lossless_cutlass_v30::launch_specialized<32, 128>(x, w, meta, out,
                                                               m, n, k, stream);
}
extern "C" int v30_m32n128_packed_decode(void const* w, void const* meta,
                                         void* out, int n, int k, int clear,
                                         void* stream) {
  return glm_lossless_cutlass_v30::launch_decode<32, 128>(w, meta, out, n, k,
                                                          clear, stream);
}
