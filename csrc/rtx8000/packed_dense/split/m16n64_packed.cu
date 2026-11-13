// SPDX-License-Identifier: Apache-2.0
#include "gemm.cuh"
#include "iterator_decode.cuh"
extern "C" int m16n64_packed_configure(int* stats) {
  return glm_packed_splitk_v2::Types<16, 64, true>::configure(stats);
}
extern "C" int m16n64_packed(void const* x, void const* w, void const* meta,
                             void* out, void* workspace, int m, int n, int k,
                             int splits, void* stream) {
  return glm_packed_splitk_v2::launch<16, 64, true>(x, w, meta, out, workspace,
                                                    m, n, k, splits, stream);
}
extern "C" int m16n64_packed_decode(void const* w, void const* meta, void* out,
                                    int n, int k, int clear, void* stream) {
  return glm_packed_splitk_v2::launch_decode<16, 64>(w, meta, out, n, k, clear,
                                                     stream);
}
