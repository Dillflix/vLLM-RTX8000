// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Weight rows are contiguous; each warp produces a column of up to 20 tokens.
// Keep FP32 accumulation and the original FP16 weights.
template <int M>
__global__ void gemv(const half* __restrict__ x, const half* __restrict__ w,
                     half* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (n >= N) return;
  float sums[M] = {};
  for (int k = lane * 8; k < K; k += 256) {
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
    for (int step = 16; step; step /= 2)
      sums[m] += __shfl_down_sync(0xffffffff, sums[m], step);
    if (!lane) y[m * N + n] = __float2half(sums[m]);
  }
}

// New token-count instantiations compile independently of the CUTLASS unit.
#define RTX_DEFINE_DENSE_GEMV_EXTRA(M)                                        \
  extern "C" int dense75_gemv_m##M(const void* x, const void* weight,         \
                                   void* output, int N, int K, int variant,   \
                                   void* stream_ptr) {                        \
    const int threads = variant == 0 ? 128 : 256;                             \
    auto stream = static_cast<cudaStream_t>(stream_ptr);                      \
    gemv<M><<<(N + threads / 32 - 1) / (threads / 32), threads, 0, stream>>>( \
        static_cast<const half*>(x), static_cast<const half*>(weight),        \
        static_cast<half*>(output), N, K);                                    \
    return int(cudaGetLastError());                                           \
  }
