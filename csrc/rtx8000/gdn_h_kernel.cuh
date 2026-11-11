// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gdn_h_launch.h"

namespace gdn_h75 {
namespace wmma = nvcuda::wmma;

struct alignas(32) Shared {
  half wk[64 * 128];
  half state_half[16 * 128];
  half residual[64 * 16];
  float temporary[16 * 128];
  float gates[64];
  float decay;
};

__device__ __forceinline__ float gate_exp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

__device__ __forceinline__ int64_t index(const void* p, int i, bool wide) {
  return wide ? static_cast<const int64_t*>(p)[i]
              : static_cast<const int*>(p)[i];
}

template <int Values>
__global__ __launch_bounds__(128) void state(
    const half* __restrict__ k, const half* __restrict__ w,
    const half* __restrict__ u, const float* __restrict__ g,
    const float* __restrict__ h0, const void* cu_seqlens,
    const void* chunk_offsets, half* __restrict__ h, half* __restrict__ v_new,
    float* __restrict__ ht, GdnH75Shape shape) {
  __shared__ Shared shared;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int head = blockIdx.y % 24;
  const int sequence = blockIdx.y / 24;
  const int first_value = blockIdx.x * Values;
  const int64_t bos = index(cu_seqlens, sequence, shape.sequences64);
  const int64_t eos = index(cu_seqlens, sequence + 1, shape.sequences64);
  const int64_t first_chunk = index(chunk_offsets, sequence, shape.chunks64);
  if (bos < 0 || eos < bos || eos > shape.tokens || first_chunk < 0 ||
      first_chunk > shape.chunks ||
      (eos - bos + 63) / 64 > shape.chunks - first_chunk)
    return;
  const int64_t state_base = (int64_t(sequence) * 24 + head) * 128 * 128;

  using RowA =
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
  using ColA =
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>;
  using RowB =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
  using ColB =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
  using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  // A V8 CTA pads its private WMMA rows; no padded row reaches global memory.
  for (int i = thread; i < 16 * 128; i += 128) {
    const int row = i / 128, col = i % 128;
    shared.temporary[i] =
        row < Values ? h0[state_base + (first_value + row) * 128 + col] : 0.f;
  }
  __syncthreads();
  Acc live[2];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    wmma::load_matrix_sync(live[j], shared.temporary + warp * 32 + j * 16, 128,
                           wmma::mem_row_major);

  for (int64_t start = bos, chunk = first_chunk; start < eos;
       start += 64, ++chunk) {
    const int count = static_cast<int>(min(int64_t(64), eos - start));
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(shared.temporary + warp * 32 + j * 16, live[j],
                              128, wmma::mem_row_major);
    __syncthreads();
    for (int i = thread; i < 16 * 128; i += 128) {
      const int row = i / 128, col = i % 128;
      const half rounded = __float2half_rn(shared.temporary[i]);
      shared.state_half[i] = rounded;
      if (row < Values)
        h[((chunk * 24 + head) * 128 + first_value + row) * 128 + col] =
            rounded;
    }
    for (int i = thread; i < 64 * 16; i += 128) {
      const int token = i / 16, col = (i % 16) * 8;
      int4 value = make_int4(0, 0, 0, 0);
      if (token < count)
        value = *reinterpret_cast<const int4*>(
            w + ((start + token) * 24 + head) * 128 + col);
      reinterpret_cast<int4*>(shared.wk)[i] = value;
    }
    if (thread < 64) {
      const float last = g[(start + count - 1) * 24 + head];
      shared.gates[thread] =
          thread < count ? gate_exp(last - g[(start + thread) * 24 + head])
                         : 0.f;
      if (thread == 0) shared.decay = gate_exp(last);
    }
    __syncthreads();

    // Preserve the source's two independent K64 dots and FP32 sum.
    Acc prediction[2];
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      wmma::fill_fragment(prediction[group], 0.f);
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        const int col = group * 64 + step * 16;
        RowA a;
        ColB b;
        wmma::load_matrix_sync(a, shared.wk + warp * 16 * 128 + col, 128);
        wmma::load_matrix_sync(b, shared.state_half + col, 128);
        wmma::mma_sync(prediction[group], a, b, prediction[group]);
      }
    }
#pragma unroll
    for (int i = 0; i < Acc::num_elements; ++i)
      prediction[0].x[i] += prediction[1].x[i];
    wmma::store_matrix_sync(shared.temporary + warp * 16 * 16, prediction[0],
                            16, wmma::mem_row_major);
    __syncthreads();
    for (int i = thread; i < 64 * 16; i += 128) {
      const int token = i / 16, value = i % 16;
      half weighted = __float2half_rn(0.f);
      if (token < count && value < Values) {
        const int64_t offset =
            ((start + token) * 24 + head) * 128 + first_value + value;
        const float residual = __half2float(u[offset]) - shared.temporary[i];
        v_new[offset] = __float2half_rn(residual);
        // Gate the unrounded residual, then round for the state-update MMA.
        weighted = __float2half_rn(residual * shared.gates[token]);
      }
      shared.residual[i] = weighted;
    }
    // The W arena is dead after prediction; reuse it for K, keeping two CTAs
    // within Turing's 64-KiB shared-memory budget.
    for (int i = thread; i < 64 * 16; i += 128) {
      const int token = i / 16, col = (i % 16) * 8;
      int4 value = make_int4(0, 0, 0, 0);
      if (token < count)
        value = *reinterpret_cast<const int4*>(
            k + ((start + token) * 8 + head / 3) * 128 + col);
      reinterpret_cast<int4*>(shared.wk)[i] = value;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      Acc update;
      wmma::fill_fragment(update, 0.f);
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        ColA a;
        RowB b;
        wmma::load_matrix_sync(a, shared.residual + step * 16 * 16, 16);
        wmma::load_matrix_sync(
            b, shared.wk + step * 16 * 128 + warp * 32 + j * 16, 128);
        wmma::mma_sync(update, a, b, update);
      }
#pragma unroll
      for (int i = 0; i < Acc::num_elements; ++i)
        live[j].x[i] =
            __fadd_rn(__fmul_rn(live[j].x[i], shared.decay), update.x[i]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 2; ++j)
    wmma::store_matrix_sync(shared.temporary + warp * 32 + j * 16, live[j], 128,
                            wmma::mem_row_major);
  __syncthreads();
  for (int i = thread; i < Values * 128; i += 128)
    ht[state_base + first_value * 128 + i] = shared.temporary[i];
}

template <int Values>
int launch(const void* k, const void* w, const void* u, const float* g,
           const float* h0, const void* cu_seqlens, const void* chunk_offsets,
           void* h, void* v_new, float* ht, GdnH75Shape shape,
           cudaStream_t stream) {
  state<Values><<<dim3(128 / Values, shape.sequences * 24), 128, 0, stream>>>(
      static_cast<const half*>(k), static_cast<const half*>(w),
      static_cast<const half*>(u), g, h0, cu_seqlens, chunk_offsets,
      static_cast<half*>(h), static_cast<half*>(v_new), ht, shape);
  return cudaGetLastError();
}
}  // namespace gdn_h75
