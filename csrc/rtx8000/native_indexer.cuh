// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Experimental GLM H32/D128 scorer and exact bounded top-512.
// The chunk/top-k/tree principle is also used by Dillflix/ds4 (MIT); this
// implementation preserves the deployed GLM Triton arithmetic, not DS4's.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>
#include <type_traits>

namespace glm_native_indexer {
constexpr int threads = 128;
constexpr int keep = 512;

__device__ __forceinline__ int dot_dimension(int instruction, int lane_quad) {
  // The large-K operand lowering pairs K0 with K16 before visiting K2/K18.
  return (instruction / 4) * 32 + (instruction % 2) * 16 +
         ((instruction % 4) / 2) * 2 + lane_quad * 4;
}

__device__ __forceinline__ uint16_t decode_e4(uint8_t b) {
  const uint32_t m = b & 127, s = (b & 128) << 8;
  if (m == 127) return uint16_t(s | 0x7e00);
  if (m < 8) {
    const float x = float(m) * (1.f / 512.f);
    return __half_as_ushort(__float2half_rn(s ? -x : x));
  }
  return uint16_t(s | ((m + 64) << 7));
}

__device__ __forceinline__ uint32_t query_pair(const uint8_t* p) {
  return uint32_t(decode_e4(p[0])) | (uint32_t(decode_e4(p[1])) << 16);
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1,
                                    uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b));
}

struct Query {
  uint32_t a[4][16];
  float w[4];
  __device__ __forceinline__ Query(const uint8_t* q, const float* weights) {
    const int g = (threadIdx.x & 31) >> 2, t = threadIdx.x & 3;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      w[h] = weights[g + h * 8];
#pragma unroll
      for (int d = 0; d < 16; ++d)
        // Match the FP8-origin Triton kWidth4 operand's actual MMA order.
        a[h][d] = query_pair(q + (g + h * 8) * 128 + dot_dimension(d, t));
    }
  }
};

// One warp scores 16 keys in two eight-column groups separated by32. This
// matches Triton's [1,4]-warp N8 layout, including its column-specific FMA.
// Reduction/FMA order matches the pinned production
// _prefill_logits_dynamic H32/D128/BN64 FP8-query, FP16-key specialization.
template <typename Key>
__device__ __forceinline__ uint32_t key_pair(const Key* p) {
  if constexpr (std::is_same_v<Key, uint8_t>)
    return query_pair(p);
  else
    return *reinterpret_cast<const uint32_t*>(p);
}

template <typename Key, bool FP8Reduction = std::is_same_v<Key, uint8_t>>
__device__ __forceinline__ void score64(const Query& q, const Key* keys,
                                        const float* scales, int start,
                                        int valid_count, int relative,
                                        float* destination) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int base = relative + warp * 8;
  float c[16] = {};
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    uint32_t b0 = 0, b1 = 0;
    if (base + g < valid_count)
      b0 = key_pair(keys + int64_t(start + base + g) * 128 +
                    dot_dimension(d, t));
    if (base + g + 32 < valid_count)
      b1 = key_pair(keys + int64_t(start + base + g + 32) * 128 +
                    dot_dimension(d, t));
    mma(c, q.a[0][d], q.a[1][d], b0);
    mma(c + 4, q.a[0][d], q.a[1][d], b1);
    mma(c + 8, q.a[2][d], q.a[3][d], b0);
    mma(c + 12, q.a[2][d], q.a[3][d], b1);
  }
  float s[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int col = base + (j >> 1) * 32 + t * 2 + (j & 1);
    s[j] = col < valid_count ? scales[start + col] : 0.f;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int col = ((j & 7) >> 2) * 2 + (j & 1);
    c[j] = fmaxf(__fmul_rn(c[j], s[col]), 0.f);
  }
  float v[4];
  if constexpr (FP8Reduction) {
    // The pinned FP8-key lowering contracts the lower head of each pair.
    // Its FP16-key specialization has different column-specific contractions.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = (j >> 1) * 4 + (j & 1);
      v[j] =
          __fadd_rn(__fmaf_rn(c[i], q.w[0], __fmul_rn(c[i + 2], q.w[1])),
                    __fmaf_rn(c[i + 8], q.w[2], __fmul_rn(c[i + 10], q.w[3])));
    }
  } else {
    v[0] = __fadd_rn(__fmaf_rn(c[2], q.w[1], __fmul_rn(c[0], q.w[0])),
                     __fmaf_rn(c[8], q.w[2], __fmul_rn(c[10], q.w[3])));
    v[1] = __fadd_rn(__fmaf_rn(c[1], q.w[0], __fmul_rn(c[3], q.w[1])),
                     __fmaf_rn(c[9], q.w[2], __fmul_rn(c[11], q.w[3])));
    v[2] = __fadd_rn(__fmaf_rn(c[12], q.w[2], __fmul_rn(c[14], q.w[3])),
                     __fmaf_rn(c[4], q.w[0], __fmul_rn(c[6], q.w[1])));
    v[3] = __fadd_rn(__fmaf_rn(c[13], q.w[2], __fmul_rn(c[15], q.w[3])),
                     __fmaf_rn(c[7], q.w[1], __fmul_rn(c[5], q.w[0])));
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __fadd_rn(v[j], __shfl_xor_sync(0xffffffff, v[j], 16));
    v[j] = __fadd_rn(v[j], __shfl_xor_sync(0xffffffff, v[j], 8));
    v[j] = __fadd_rn(v[j], __shfl_xor_sync(0xffffffff, v[j], 4));
    if (g == 0) {
      int offset = warp * 8 + (j >> 1) * 32 + t * 2 + (j & 1);
      destination[offset] =
          relative + offset < valid_count ? v[j] : -CUDART_INF_F;
    }
  }
}

__device__ __forceinline__ int row_length(int start, int end, int n,
                                          int width) {
  return start < 0 ? 0 : max(0, min(width, min(end, n) - start));
}

template <typename Key, bool FP8Reduction = std::is_same_v<Key, uint8_t>>
__global__ void score_debug(const uint8_t* q, const Key* keys,
                            const float* scales, const float* weights,
                            const int* starts, const int* ends, float* out,
                            int n, int width) {
  const int row = blockIdx.x, rel = blockIdx.y * 64;
  const int start = starts[row];
  const int length = row_length(start, ends[row], n, width);
  __shared__ float scratch[64];
  if (rel < length) {
    Query query(q + int64_t(row) * 4096, weights + row * 32);
    score64<Key, FP8Reduction>(query, keys, scales, start, length, rel,
                               scratch);
  } else if (threadIdx.x < 64)
    scratch[threadIdx.x] = -CUDART_INF_F;
  __syncthreads();
  if (threadIdx.x < 64 && rel + threadIdx.x < width)
    out[int64_t(row) * width + rel + threadIdx.x] = scratch[threadIdx.x];
}

// Descending score, ascending index. Zero is the invalid padding key. Both
// signed zeros compare equal; finite scores and infinities are supported.
__device__ __forceinline__ uint64_t rank_key(float score, int index) {
  uint32_t u = __float_as_uint(score == 0.f ? 0.f : score);
  uint32_t ordered = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
  return (uint64_t(ordered) << 32) | (0xffffffffu - uint32_t(index));
}

template <int Partition>
struct SortStorage {
  using Sort = cub::BlockRadixSort<uint64_t, threads, Partition / threads>;
  union Storage {
    float scores[Partition];
    typename Sort::TempStorage sort;
  };
};

template <int Partition, bool ShortCausal = false, typename Key = __half,
          bool FP8Reduction = std::is_same_v<Key, uint8_t>>
__global__ void fused_local(const uint8_t* q, const Key* keys,
                            const float* scales, const float* weights,
                            const int* starts, const int* ends,
                            uint64_t* candidates, int n, int width, int parts) {
  const int row = blockIdx.x, part = blockIdx.y, begin = part * Partition;
  const int start = starts[row];
  const int length = row_length(start, ends[row], n, width);
  uint64_t* dst = candidates + (int64_t(row) * parts + part) * keep;
  if (begin >= length) {
    for (int i = threadIdx.x; i < keep; i += threads) dst[i] = 0;
    return;
  }
  using S = SortStorage<Partition>;
  using Sort = typename S::Sort;
  __shared__ typename S::Storage sm;
  {
    Query query(q + int64_t(row) * 4096, weights + row * 32);
    for (int tile = 0; tile < Partition; tile += 64) {
      if (begin + tile < length)
        score64<Key, FP8Reduction>(query, keys, scales, start, length,
                                   begin + tile, sm.scores + tile);
      else if (threadIdx.x < 64)
        sm.scores[tile + threadIdx.x] = -CUDART_INF_F;
    }
  }
  __syncthreads();
  uint64_t ranked[Partition / threads];
#pragma unroll
  for (int i = 0; i < Partition / threads; ++i) {
    const int col = threadIdx.x * (Partition / threads) + i;
    ranked[i] =
        begin + col < length
            ? rank_key(ShortCausal && length <= keep ? 0.f : sm.scores[col],
                       begin + col)
            : 0;
  }
  __syncthreads();
  Sort(sm.sort).SortDescending(ranked);
#pragma unroll
  for (int i = 0; i < Partition / threads; ++i) {
    const int col = threadIdx.x * (Partition / threads) + i;
    if (col < keep) dst[col] = ranked[i];
  }
}

template <int Partition>
__global__ void merge(const uint64_t* src, uint64_t* dst, int* indices,
                      int sets, int next_sets) {
  using S = SortStorage<Partition>;
  using Sort = typename S::Sort;
  __shared__ typename Sort::TempStorage temp;
  const int row = blockIdx.x, group = blockIdx.y;
  const int begin = group * Partition;
  uint64_t ranked[Partition / threads];
#pragma unroll
  for (int i = 0; i < Partition / threads; ++i) {
    const int col = threadIdx.x * (Partition / threads) + i;
    ranked[i] = begin + col < sets * keep
                    ? src[int64_t(row) * sets * keep + begin + col]
                    : 0;
  }
  Sort(temp).SortDescending(ranked);
#pragma unroll
  for (int i = 0; i < Partition / threads; ++i) {
    const int col = threadIdx.x * (Partition / threads) + i;
    if (col < keep) {
      if (indices)
        indices[int64_t(row) * keep + col] =
            ranked[i] ? int(0xffffffffu - uint32_t(ranked[i])) : -1;
      else
        dst[(int64_t(row) * next_sets + group) * keep + col] = ranked[i];
    }
  }
}
}  // namespace glm_native_indexer
