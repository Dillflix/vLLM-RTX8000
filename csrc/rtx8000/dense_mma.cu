// SPDX-License-Identifier: Apache-2.0
// The shared plan list drives declarations, dispatch, and CMake source
// selection.
#define RTX_DENSE_MMA_VARIANT(ID, NAME, BN, BK, SKEW)                      \
  extern "C" int dense75_##NAME(const void*, const void*, void*, int, int, \
                                int, void*);
#include "dense_mma_variants.inc"
#undef RTX_DENSE_MMA_VARIANT

extern "C" int dense75_mma_launch(const void* x, const void* weight,
                                  void* output, int M, int N, int K,
                                  int variant, void* stream_ptr) {
  if (M < 1 || M > 16 || K % 8) return -1;
  switch (variant) {
#define RTX_DENSE_MMA_VARIANT(ID, NAME, BN, BK, SKEW) \
  case ID:                                            \
    return dense75_##NAME(x, weight, output, M, N, K, stream_ptr);
#include "dense_mma_variants.inc"
#undef RTX_DENSE_MMA_VARIANT
    default:
      return -1;
  }
}
