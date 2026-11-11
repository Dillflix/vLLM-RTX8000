// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "launch.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

namespace qsa75_tuned {
using namespace nvcuda;

#define RTX_QSA_VARIANT(ID, NAME, KEYS, WARPS, PREFETCH) \
  struct NAME {                                          \
    static constexpr int keys = KEYS, warps = WARPS;     \
    static constexpr bool prefetch = PREFETCH;           \
  };
#include "qsa_variants.inc"
#undef RTX_QSA_VARIANT

template <typename Plan>
__global__ void attention(const half* q, const half* packed, const int* indices,
                          const int* table, const int* requests,
                          const half* gate, half* output, float* partial,
                          float* lse, Qsa75Shape shape) {
  constexpr int BK = Plan::keys, NW = Plan::warps, NT = NW * 32;
  constexpr int COLUMNS = BK / 32, ACCUMULATORS = 16 / NW;
  __shared__ __align__(32) half sq[16 * 256];
  __shared__ __align__(32) half kv[BK * 256];
  __shared__ __align__(32) float score[16 * BK];
  __shared__ __align__(32) half prob[16 * BK];
  __shared__ __align__(32) float alpha_matrix[16 * 16];
  __shared__ float maximum[16], denominator[16];
  __shared__ long long offsets[BK];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int row = blockIdx.x, split = blockIdx.y, M = shape.rows;
  const int req = requests[row];
  const int count =
      min(shape.topk, max(0, indices[row * shape.index_stride + shape.topk]));
  for (int i = tid; i < 16 * 256; i += NT)
    sq[i] = i / 256 < 12 ? q[row * shape.query_stride + i] : __float2half(0.f);
  if (tid < 16) {
    maximum[tid] = -CUDART_INF_F;
    denominator[tid] = 0.f;
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[ACCUMULATORS];
#pragma unroll
  for (int z = 0; z < ACCUMULATORS; ++z) wmma::fill_fragment(acc[z], 0.f);
  __syncthreads();
  for (int tile = split; tile < (count + BK - 1) / BK; tile += shape.splits) {
    if (tid < BK) {
      int column = tile * BK + tid;
      int logical =
          column < count ? indices[row * shape.index_stride + column] : -1;
      int pg = logical >= 0 ? logical / shape.page_size : -1;
      int physical = (req >= 0 && req < shape.requests && pg >= 0 &&
                      pg < shape.table_width)
                         ? table[req * shape.table_stride + pg]
                         : -1;
      offsets[tid] = (physical >= 0 && physical < shape.pages)
                         ? (long long)physical * shape.cache_block_stride +
                               (logical % shape.page_size) * 512
                         : -1;
    }
    __syncthreads();
    int4 prefetched[BK * 32 / NT];
#pragma unroll
    for (int i = 0; i < BK * 32 / NT; ++i) {
      int vec = tid + i * NT;
      long long off = offsets[vec / 32];
      int4 value = {0, 0, 0, 0};
      if (off >= 0)
        value = *reinterpret_cast<const int4*>(packed + off + (vec % 32) * 8);
      reinterpret_cast<int4*>(kv)[vec] = value;
      if constexpr (Plan::prefetch) {
        prefetched[i] = {0, 0, 0, 0};
        if (off >= 0)
          prefetched[i] = *reinterpret_cast<const int4*>(packed + off + 256 +
                                                         (vec % 32) * 8);
      }
    }
    __syncthreads();
    if (warp < BK / 16) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
#pragma unroll
      for (int k = 0; k < 256; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, sq + k, 256);
        wmma::load_matrix_sync(b, kv + warp * 16 * 256 + k, 256);
        wmma::mma_sync(s, a, b, s);
      }
      wmma::store_matrix_sync(score + warp * 16, s, BK, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int h = warp; h < 16; h += NW) {
      float s[COLUMNS], mx = -CUDART_INF_F;
#pragma unroll
      for (int c = 0; c < COLUMNS; ++c) {
        int column = lane + c * 32;
        s[c] = offsets[column] >= 0
                   ? score[h * BK + column] * (1.4426950408889634f / 16.f)
                   : -CUDART_INF_F;
        mx = fmaxf(mx, s[c]);
      }
#pragma unroll
      for (int d = 16; d; d /= 2)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
      float next = fmaxf(mx, maximum[h]);
      float alpha = isfinite(next) ? exp2f(maximum[h] - next) : 0.f;
      float sum = 0.f;
#pragma unroll
      for (int c = 0; c < COLUMNS; ++c) {
        float p = isfinite(s[c]) ? exp2f(s[c] - next) : 0.f;
        prob[h * BK + lane + c * 32] = __float2half(p);
        sum += p;
      }
#pragma unroll
      for (int d = 16; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
      if (lane < 16) alpha_matrix[h * 16 + lane] = alpha;
      // Independent thread scheduling requires a shared-memory ordering point
      // before lane 0 replaces the maximum that all lanes just consumed.
      __syncwarp();
      if (!lane) {
        maximum[h] = next;
        denominator[h] = denominator[h] * alpha + sum;
      }
    }
#pragma unroll
    for (int i = 0; i < BK * 32 / NT; ++i) {
      int vec = tid + i * NT;
      int4 value = {0, 0, 0, 0};
      if constexpr (Plan::prefetch) {
        value = prefetched[i];
      } else {
        long long off = offsets[vec / 32];
        if (off >= 0)
          value = *reinterpret_cast<const int4*>(packed + off + 256 +
                                                 (vec % 32) * 8);
      }
      reinterpret_cast<int4*>(kv)[vec] = value;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> alpha;
    wmma::load_matrix_sync(alpha, alpha_matrix, 16, wmma::mem_row_major);
#pragma unroll
    for (int z = 0; z < ACCUMULATORS; ++z) {
#pragma unroll
      for (int i = 0; i < acc[z].num_elements; ++i) acc[z].x[i] *= alpha.x[i];
#pragma unroll
      for (int k = 0; k < BK; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
        wmma::load_matrix_sync(a, prob + k, BK);
        wmma::load_matrix_sync(b, kv + k * 256 + (warp + z * NW) * 16, 256);
        wmma::mma_sync(acc[z], a, b, acc[z]);
      }
    }
    __syncthreads();
  }
  float* result = reinterpret_cast<float*>(kv);
#pragma unroll
  for (int z = 0; z < ACCUMULATORS; ++z)
    wmma::store_matrix_sync(result + (warp + z * NW) * 16, acc[z], 256,
                            wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 12 * 256; i += NT) {
    float value =
        denominator[i / 256] > 0 ? result[i] / denominator[i / 256] : 0.f;
    if (shape.splits == 1) {
      value = __half2float(__float2half(value));
      output[row * shape.output_stride + i] = __float2half(
          value /
          (1.f + expf(-__half2float(gate[row * shape.gate_stride + i]))));
    } else {
      partial[((long long)split * M + row) * 3072 + i] = value;
    }
  }
  if (shape.splits > 1 && tid < 12)
    lse[(split * M + row) * 12 + tid] =
        denominator[tid] > 0 ? maximum[tid] + log2f(denominator[tid])
                             : -CUDART_INF_F;
}
}  // namespace qsa75_tuned

#define RTX_DEFINE_QSA(NAME)                                                   \
  extern "C" int sparse75_##NAME(                                              \
      const void* q, const void* kv, const void* idx, const void* table,       \
      const void* req, const void* gate, void* out, void* partial, void* lse,  \
      Qsa75Shape shape, void* stream) {                                        \
    using Plan = qsa75_tuned::NAME;                                            \
    qsa75_tuned::attention<Plan>                                               \
        <<<dim3(shape.rows, shape.splits), Plan::warps * 32, 0,                \
           static_cast<cudaStream_t>(stream)>>>(                               \
            static_cast<const half*>(q), static_cast<const half*>(kv),         \
            static_cast<const int*>(idx), static_cast<const int*>(table),      \
            static_cast<const int*>(req), static_cast<const half*>(gate),      \
            static_cast<half*>(out), static_cast<float*>(partial),             \
            static_cast<float*>(lse), shape);                                  \
    return int(cudaGetLastError());                                            \
  }                                                                            \
  extern "C" int sparse75_##NAME##_resources(int* regs, int* shared,           \
                                             int* blocks) {                    \
    using Plan = qsa75_tuned::NAME;                                            \
    cudaFuncAttributes attrs;                                                  \
    auto status = cudaFuncGetAttributes(&attrs, qsa75_tuned::attention<Plan>); \
    if (status != cudaSuccess) return int(status);                             \
    *regs = attrs.numRegs;                                                     \
    *shared = attrs.sharedSizeBytes;                                           \
    return int(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                  \
        blocks, qsa75_tuned::attention<Plan>, Plan::warps * 32, 0));           \
  }
