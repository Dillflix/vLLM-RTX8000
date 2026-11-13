// SPDX-License-Identifier: Apache-2.0
#include "gemv_mixed.cuh"
extern "C" int glm_mixed_gemv_m1(const void* x, const void* p, const void* meta,
                                 void* out, int N, int K, int variant,
                                 void* sp) {
  const int threads = variant == 0 ? 128 : 256;
  glm_mixed_gemv_v4::packed_gemv<1>
      <<<(N + threads / 16 - 1) / (threads / 16), threads, 0,
         static_cast<cudaStream_t>(sp)>>>(
          static_cast<const half*>(x), static_cast<const unsigned char*>(p),
          static_cast<const unsigned long long*>(meta), static_cast<half*>(out),
          N, K);
  return int(cudaGetLastError());
}
extern "C" int glm_plain_gemv_m1(const void* x, const void* w, void* out, int N,
                                 int K, int variant, void* sp) {
  const int threads = variant == 0 ? 128 : 256;
  glm_plain_gemv_v4::gemv<1><<<(N + threads / 16 - 1) / (threads / 16), threads,
                               0, static_cast<cudaStream_t>(sp)>>>(
      static_cast<const half*>(x), static_cast<const half*>(w),
      static_cast<half*>(out), N, K);
  return int(cudaGetLastError());
}
