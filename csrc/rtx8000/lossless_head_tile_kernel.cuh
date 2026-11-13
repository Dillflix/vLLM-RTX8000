// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Isolated exact arithmetic12/plain scheduling pair. Production is unchanged.
namespace head_mma75_12_tileplanes_v1 {
constexpr int BK = 64;
constexpr int THREADS = 128;

__device__ __forceinline__ int swizzle(int row, int column) {
  return row * BK + (column ^ ((row & 7) * 8));
}

__device__ __forceinline__ int4 load128(const half* pointer) {
  int4 value;
  // Keep prefetch loads before the later volatile MMA sequence. This is an
  // ordinary synchronous global load; SM75 has no cp.async dependency here.
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer)
               : "memory");
  return value;
}

__device__ __forceinline__ void load_encoded16(const unsigned char* low,
                                               const unsigned char* high,
                                               int4& first, int4& second) {
  unsigned l0, l1, l2, l3, h0, h1;
  asm volatile(
      "ld.global.v4.u32 {%0,%1,%2,%3}, [%6];\n"
      "ld.global.v2.u32 {%4,%5}, [%7];"
      : "=r"(l0), "=r"(l1), "=r"(l2), "=r"(l3), "=r"(h0), "=r"(h1)
      : "l"(low), "l"(high)
      : "memory");
  const unsigned lows[4] = {l0, l1, l2, l3};
  const unsigned highs[2] = {h0, h1};
  auto a = reinterpret_cast<unsigned short*>(&first);
  auto b = reinterpret_cast<unsigned short*>(&second);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const unsigned code = ((lows[i / 4] >> ((i & 3) * 8)) & 255) |
                          (((highs[i / 8] >> ((i & 7) * 4)) & 15) << 8);
    const unsigned payload = code & 2047;
    const int shift = min(3, max(0, int(payload >> 7) - 1));
    const unsigned short bits =
        ((payload - (shift << 7)) << shift) | ((code & 2048) << 4);
    if (i < 8)
      a[i] = bits;
    else
      b[i - 8] = bits;
  }
}

template <int BN, bool PACKED>
__device__ __forceinline__ void prefetch(const half* x, const void* weight,
                                         int M, int N, int K, int first_n,
                                         int first_k, int4& input,
                                         int4 (&weights)[BN / 16]) {
  static_assert(!PACKED || BN == 64, "N64 tile format");
  const int tid = threadIdx.x;
  const int row = tid / 8;
  const int column = (tid & 7) * 8;
  input = row < M ? load128(x + size_t(row) * K + first_k + column)
                  : make_int4(0, 0, 0, 0);
#pragma unroll
  for (int i = 0; i < BN / 32; ++i) {
    const int vector = tid + i * THREADS;
    const int n = first_n + vector / 4;
    const int k = first_k + (vector & 3) * 16;
    if (n >= N)
      weights[2 * i] = weights[2 * i + 1] = make_int4(0, 0, 0, 0);
    else if constexpr (PACKED) {
      const auto bytes = static_cast<const unsigned char*>(weight);
      // One contiguous low/high tile, with exact-sized N-tail storage.
      const int tile_rows = min(BN, N - first_n);
      const size_t tile_offset =
          (size_t(first_n) * K + size_t(first_k) * tile_rows) * 3 / 2;
      const auto tile = bytes + tile_offset;
      load_encoded16(tile + vector * 16, tile + tile_rows * BK + vector * 8,
                     weights[2 * i], weights[2 * i + 1]);
    } else {
      const auto values = static_cast<const half*>(weight) + size_t(n) * K + k;
      weights[2 * i] = load128(values);
      weights[2 * i + 1] = load128(values + 8);
    }
  }
}

template <int BN>
__device__ __forceinline__ void stage(half* target, const int4& input,
                                      const int4 (&weights)[BN / 16]) {
  const int tid = threadIdx.x;
  *reinterpret_cast<int4*>(target + swizzle(tid / 8, (tid & 7) * 8)) = input;
  half* b = target + 16 * BK;
#pragma unroll
  for (int i = 0; i < BN / 32; ++i) {
    const int vector = tid + i * THREADS;
    const int row = vector / 4;
    const int column = (vector & 3) * 16;
    *reinterpret_cast<int4*>(b + swizzle(row, column)) = weights[2 * i];
    *reinterpret_cast<int4*>(b + swizzle(row, column + 8)) = weights[2 * i + 1];
  }
}

__device__ __forceinline__ void mma(unsigned a0, unsigned a1, unsigned b,
                                    float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b));
}

template <int BN>
__device__ __forceinline__ void compute(const half* source,
                                        float (&acc)[BN / 32][4]) {
  constexpr int NF = BN / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int row = lane >> 2;
  const int pair = (lane & 3) * 2;
  const half* b = source + 16 * BK;
#pragma unroll
  for (int k = 0; k < BK; k += 8) {
    const unsigned a0 =
        *reinterpret_cast<const unsigned*>(source + swizzle(row, k + pair));
    const unsigned a1 =
        *reinterpret_cast<const unsigned*>(source + swizzle(row + 8, k + pair));
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      const int n = (warp * NF + f) * 8 + row;
      const unsigned br =
          *reinterpret_cast<const unsigned*>(b + swizzle(n, k + pair));
      mma(a0, a1, br, acc[f]);
    }
  }
}

template <int BN, int STAGES, bool PACKED>
__global__ void head_unpacked(const half* __restrict__ x,
                              const void* __restrict__ weight,
                              half* __restrict__ output, int M, int N, int K) {
  static_assert(BN == 64 || BN == 128);
  static_assert(STAGES == 1 || STAGES == 2);
  constexpr int NF = BN / 32;
  constexpr int STAGE_VALUES = (16 + BN) * BK;
  __shared__ __align__(32) half tiles[STAGES * STAGE_VALUES];
  float acc[NF][4] = {};
  const int first_n = blockIdx.x * BN;
  int4 input;
  int4 weights[BN / 16];
  if constexpr (STAGES == 1) {
    for (int k = 0; k < K; k += BK) {
      prefetch<BN, PACKED>(x, weight, M, N, K, first_n, k, input, weights);
      stage<BN>(tiles, input, weights);
      __syncthreads();
      compute<BN>(tiles, acc);
      __syncthreads();
    }
  } else {
    prefetch<BN, PACKED>(x, weight, M, N, K, first_n, 0, input, weights);
    stage<BN>(tiles, input, weights);
    __syncthreads();
    int active = 0;
    for (int k = 0; k < K; k += BK) {
      if (k + BK < K)
        prefetch<BN, PACKED>(x, weight, M, N, K, first_n, k + BK, input,
                             weights);
      compute<BN>(tiles + active * STAGE_VALUES, acc);
      if (k + BK < K)
        stage<BN>(tiles + (active ^ 1) * STAGE_VALUES, input, weights);
      __syncthreads();
      active ^= 1;
    }
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  const int row = lane >> 2;
  const int pair = (lane & 3) * 2;
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    const int n = first_n + (warp * NF + f) * 8 + pair;
    if (row < M && n < N)
      output[size_t(row) * N + n] = __float2half_rn(acc[f][0]);
    if (row < M && n + 1 < N)
      output[size_t(row) * N + n + 1] = __float2half_rn(acc[f][1]);
    if (row + 8 < M && n < N)
      output[size_t(row + 8) * N + n] = __float2half_rn(acc[f][2]);
    if (row + 8 < M && n + 1 < N)
      output[size_t(row + 8) * N + n + 1] = __float2half_rn(acc[f][3]);
  }
}

template <int BN, int STAGES, bool PACKED>
int launch(const void* x, const void* weight, void* output, int M, int N, int K,
           void* stream_ptr) {
  if (M < 0 || M > 16 || N <= 0 || K <= 0 || K % BK ||
      reinterpret_cast<size_t>(x) % 16 || reinterpret_cast<size_t>(weight) % 16)
    return -1;
  if (M == 0) return 0;
  head_unpacked<BN, STAGES, PACKED><<<(N + BN - 1) / BN, THREADS, 0,
                                      static_cast<cudaStream_t>(stream_ptr)>>>(
      static_cast<const half*>(x), weight, static_cast<half*>(output), M, N, K);
  return int(cudaGetLastError());
}
}  // namespace head_mma75_12_tileplanes_v1
