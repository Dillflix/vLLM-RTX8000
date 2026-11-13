// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "load_weight8.cuh"

namespace glm_mixed_gemv_v4 {
template <int M>
__global__ void packed_gemv(const half* __restrict__ x,
                            const unsigned char* __restrict__ w,
                            const unsigned long long* __restrict__ metadata,
                            half* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 15;
  const int n = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
  if (n >= N) return;
  const int group = n / 256;
  const unsigned long long group_offset = metadata[group * 2];
  const int width = int(metadata[group * 2 + 1]);
  float sums[M] = {};
  for (int k = lane * 8; k < K; k += 128) {
    const int4 packed_w = load_weight8(w, group_offset, width, n, k, N, K);
    const half* wh = reinterpret_cast<const half*>(&packed_w);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const int4 packed_x = *reinterpret_cast<const int4*>(x + m * K + k);
      const half* xh = reinterpret_cast<const half*>(&packed_x);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sums[m] = fmaf(__half2float(wh[i]), __half2float(xh[i]), sums[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int step = 8; step; step /= 2)
      sums[m] += __shfl_down_sync(0xffffu << ((threadIdx.x & 31) / 16 * 16),
                                  sums[m], step, 16);
    if (!lane) y[m * N + n] = __float2half(sums[m]);
  }
}
}  // namespace glm_mixed_gemv_v4

namespace glm_plain_gemv_v4 {
template <int M>
__global__ void gemv(const half* __restrict__ x, const half* __restrict__ w,
                     half* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 15;
  const int n = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
  if (n >= N) return;
  float sums[M] = {};
  for (int k = lane * 8; k < K; k += 128) {
    const int4 packed_w = *reinterpret_cast<const int4*>(w + n * K + k);
    const half* wh = reinterpret_cast<const half*>(&packed_w);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const int4 packed_x = *reinterpret_cast<const int4*>(x + m * K + k);
      const half* xh = reinterpret_cast<const half*>(&packed_x);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sums[m] = fmaf(__half2float(wh[i]), __half2float(xh[i]), sums[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int step = 8; step; step /= 2)
      sums[m] += __shfl_down_sync(0xffffu << ((threadIdx.x & 31) / 16 * 16),
                                  sums[m], step, 16);
    if (!lane) y[m * N + n] = __float2half(sums[m]);
  }
}
}  // namespace glm_plain_gemv_v4
