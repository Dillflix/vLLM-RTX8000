// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace dense75 {
using namespace nvcuda;

// Verification has at most sixteen rows. A full sixteen-row WMMA tile avoids
// the unused second M tile in the earlier CUTLASS candidates. Each weight
// tile serves every live row, including the large vocabulary projection.
template <int BN, int BK, int SKEW = 0>
__global__ void skinny_mma(const half* __restrict__ x,
                           const half* __restrict__ weight,
                           half* __restrict__ output, int M, int N, int K) {
  constexpr int WARPS = 4;
  constexpr int NF = BN / (16 * WARPS);
  constexpr int LD = BK + SKEW;
  __shared__ __align__(32) half sx[16 * LD];
  __shared__ __align__(32) half sw[BN * LD];
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int first_n = blockIdx.x * BN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f) wmma::fill_fragment(acc[f], 0.f);

  for (int base_k = 0; base_k < K; base_k += BK) {
    for (int vec = tid; vec < 16 * BK / 8; vec += 128) {
      const int row = vec / (BK / 8);
      const int column = (vec % (BK / 8)) * 8;
      int4 value = {0, 0, 0, 0};
      if (row < M && base_k + column < K)
        value = *reinterpret_cast<const int4*>(x + row * K + base_k + column);
      *reinterpret_cast<int4*>(sx + row * LD + column) = value;
    }
    for (int vec = tid; vec < BN * BK / 8; vec += 128) {
      const int row = vec / (BK / 8);
      const int column = (vec % (BK / 8)) * 8;
      int4 value = {0, 0, 0, 0};
      if (first_n + row < N && base_k + column < K)
        value = *reinterpret_cast<const int4*>(weight + (first_n + row) * K +
                                               base_k + column);
      *reinterpret_cast<int4*>(sw + row * LD + column) = value;
    }
    __syncthreads();
#pragma unroll
    for (int tile_k = 0; tile_k < BK; tile_k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::load_matrix_sync(a, sx + tile_k, LD);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(b, sw + (warp + f * WARPS) * 16 * LD + tile_k,
                               LD);
        wmma::mma_sync(acc[f], a, b, acc[f]);
      }
    }
    __syncthreads();
  }

  // The weight tile is dead; its storage is large enough for the FP32 result.
  static_assert(BN * LD * sizeof(half) >= 16 * BN * sizeof(float));
  float* result = reinterpret_cast<float*>(sw);
#pragma unroll
  for (int f = 0; f < NF; ++f)
    wmma::store_matrix_sync(result + (warp + f * WARPS) * 16, acc[f], BN,
                            wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < M * BN; i += 128) {
    const int row = i / BN;
    const int column = i % BN;
    if (first_n + column < N)
      output[row * N + first_n + column] = __float2half(result[i]);
  }
}
}  // namespace dense75

namespace dense75 {
#define RTX_DENSE_MMA_VARIANT(ID, NAME, BN, BK, SKEW) \
  struct NAME {                                       \
    static constexpr int n = BN, k = BK, skew = SKEW; \
  };
#include "dense_mma_variants.inc"
#undef RTX_DENSE_MMA_VARIANT
}  // namespace dense75

#define RTX_DEFINE_DENSE_MMA(NAME)                                         \
  extern "C" int dense75_##NAME(const void* x, const void* weight,         \
                                void* output, int M, int N, int K,         \
                                void* stream_ptr) {                        \
    using Plan = dense75::NAME;                                            \
    auto stream = static_cast<cudaStream_t>(stream_ptr);                   \
    dense75::skinny_mma<Plan::n, Plan::k, Plan::skew>                      \
        <<<(N + Plan::n - 1) / Plan::n, 128, 0, stream>>>(                 \
            static_cast<const half*>(x), static_cast<const half*>(weight), \
            static_cast<half*>(output), M, N, K);                          \
    return int(cudaGetLastError());                                        \
  }
