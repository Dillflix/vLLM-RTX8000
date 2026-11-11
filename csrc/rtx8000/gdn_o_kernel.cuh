// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gdn_o_launch.h"

namespace gdn75 {
namespace wmma = nvcuda::wmma;

// One CTA owns one grouped-query head and a query subtile of a 64-token
// chunk. The ungated QK product is reused for all three value heads and
// all 128 output channels. H is stored as [value, key], as in chunk_o.py.
template <int QueryRows>
struct Shared {
  alignas(32) half q[QueryRows * 128];
  // K, then V, then FP32 output. Explicit barriers protect these lifetimes.
  alignas(32) half work[64 * 128];
  alignas(32) float score[QueryRows * 64];
  alignas(32) half probability[QueryRows * 64];
  // Loading this as an accumulator fragment scales each query row without
  // relying on the architecture-specific layout of WMMA fragment registers.
  alignas(32) float gate_matrix[QueryRows * 16];
  alignas(32) float gate[64];
};

__device__ __forceinline__ int64_t index(const void* ptr, bool wide,
                                         int64_t offset) {
  return wide ? static_cast<const int64_t*>(ptr)[offset]
              : static_cast<const int32_t*>(ptr)[offset];
}

// Match Triton's tl.exp implementation (base-2 approximate exponential),
// retaining FP32 gate arithmetic and the FP16 boundary before the PV MMA.
__device__ __forceinline__ float gate_exp(float x) {
  return exp2f(x * 1.4426950408889634f);
}

template <int QueryRows>
__global__ __launch_bounds__(256) void output(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, const half* __restrict__ h,
    const float* __restrict__ g, const void* cu_seqlens,
    const void* chunk_indices, half* __restrict__ out, Gdn75Shape shape) {
  static_assert(QueryRows == 16 || QueryRows == 32);
  constexpr int QueryTiles = QueryRows / 16;
  __shared__ Shared<QueryRows> smem;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  // Adjacent CTAs consume the same H snapshot, improving L2 reuse of the
  // directly loaded state operands across the query tiles.
  const int chunk = blockIdx.x / (64 / QueryRows);
  const int key_head = blockIdx.y;
  const int query_offset = (blockIdx.x % (64 / QueryRows)) * QueryRows;
  const int64_t sequence = index(chunk_indices, shape.indices64, 2LL * chunk);
  const int64_t local_chunk =
      index(chunk_indices, shape.indices64, 2LL * chunk + 1);
  // Metadata comes from prepare_chunk_indices. Invalid entries must never
  // cause out-of-bounds accesses, even during padded graph execution.
  if (sequence < 0 || sequence >= shape.sequences || local_chunk < 0) return;
  const int64_t begin = index(cu_seqlens, shape.indices64, sequence);
  const int64_t end = index(cu_seqlens, shape.indices64, sequence + 1);
  if (begin < 0 || end < begin || end > shape.tokens ||
      local_chunk >= (end - begin + 63) / 64)
    return;
  const int64_t chunk_begin = begin + local_chunk * 64;
  const int64_t query_begin = chunk_begin + query_offset;
  if (query_begin >= end) return;

  // Eight half elements per transaction; every supported tensor is contiguous.
  for (int i = tid; i < QueryRows * 16; i += 256) {
    const int row = i / 16;
    const int column = (i % 16) * 8;
    int4 data = make_int4(0, 0, 0, 0);
    if (query_begin + row < end)
      data = *reinterpret_cast<const int4*>(
          q + ((query_begin + row) * 8 + key_head) * 128 + column);
    *reinterpret_cast<int4*>(smem.q + row * 128 + column) = data;
  }
  for (int i = tid; i < 64 * 16; i += 256) {
    const int row = i / 16;
    const int column = (i % 16) * 8;
    int4 data = make_int4(0, 0, 0, 0);
    if (chunk_begin + row < end)
      data = *reinterpret_cast<const int4*>(
          k + ((chunk_begin + row) * 8 + key_head) * 128 + column);
    *reinterpret_cast<int4*>(smem.work + row * 128 + column) = data;
  }
  __syncthreads();

  using A = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
  using BCol =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
  using BRow =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
  using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  if (warp < QueryTiles * 4) {
    const int query_tile = warp / 4;
    const int key_tile = warp % 4;
    Acc raw;
    wmma::fill_fragment(raw, 0.0f);
#pragma unroll
    for (int kk = 0; kk < 128; kk += 16) {
      A a;
      BCol b;
      wmma::load_matrix_sync(a, smem.q + query_tile * 16 * 128 + kk, 128);
      wmma::load_matrix_sync(b, smem.work + key_tile * 16 * 128 + kk, 128);
      wmma::mma_sync(raw, a, b, raw);
    }
    wmma::store_matrix_sync(smem.score + query_tile * 16 * 64 + key_tile * 16,
                            raw, 64, wmma::mem_row_major);
  }
  __syncthreads();

#pragma unroll
  for (int grouped_head = 0; grouped_head < 3; ++grouped_head) {
    const int head = key_head * 3 + grouped_head;
    if (tid < 64)
      smem.gate[tid] =
          chunk_begin + tid < end ? g[(chunk_begin + tid) * 24 + head] : 0.0f;
    __syncthreads();
    for (int i = tid; i < QueryRows * 64; i += 256) {
      const int row = i / 64;
      const int key = i % 64;
      const int query = query_offset + row;
      float p = 0.0f;
      if (key <= query && query_begin + row < end && chunk_begin + key < end)
        p = smem.score[i] * gate_exp(smem.gate[query] - smem.gate[key]);
      smem.probability[i] = __float2half_rn(p);
    }
    for (int i = tid; i < QueryRows * 16; i += 256)
      smem.gate_matrix[i] = gate_exp(smem.gate[query_offset + i / 16]);

    // Each warp computes 16 value channels. Global H loads avoid a 32 KiB
    // shared-memory allocation, permitting two q16 CTAs per Turing SM.
    Acc state[QueryTiles];
#pragma unroll
    for (int qt = 0; qt < QueryTiles; ++qt) {
      wmma::fill_fragment(state[qt], 0.0f);
#pragma unroll
      for (int kk = 0; kk < 128; kk += 16) {
        A a;
        BCol b;
        wmma::load_matrix_sync(a, smem.q + qt * 16 * 128 + kk, 128);
        wmma::load_matrix_sync(
            b,
            h + (static_cast<int64_t>(chunk) * 24 + head) * 128 * 128 +
                warp * 16 * 128 + kk,
            128);
        wmma::mma_sync(state[qt], a, b, state[qt]);
      }
    }
    // QK has completed in every warp before K storage is reused for V.
    for (int i = tid; i < 64 * 16; i += 256) {
      const int row = i / 16;
      const int column = (i % 16) * 8;
      int4 data = make_int4(0, 0, 0, 0);
      if (chunk_begin + row < end)
        data = *reinterpret_cast<const int4*>(
            v + ((chunk_begin + row) * 24 + head) * 128 + column);
      *reinterpret_cast<int4*>(smem.work + row * 128 + column) = data;
    }
    __syncthreads();

#pragma unroll
    for (int qt = 0; qt < QueryTiles; ++qt) {
      Acc local;
      wmma::fill_fragment(local, 0.0f);
#pragma unroll
      for (int kk = 0; kk < 64; kk += 16) {
        A a;
        BRow b;
        wmma::load_matrix_sync(a, smem.probability + qt * 16 * 64 + kk, 64);
        wmma::load_matrix_sync(b, smem.work + kk * 128 + warp * 16, 128);
        wmma::mma_sync(local, a, b, local);
      }
      Acc gate;
      wmma::load_matrix_sync(gate, smem.gate_matrix + qt * 16 * 16, 16,
                             wmma::mem_row_major);
#pragma unroll
      for (int i = 0; i < Acc::num_elements; ++i)
        state[qt].x[i] = (state[qt].x[i] * gate.x[i]) * shape.scale +
                         local.x[i] * shape.scale;
    }
    __syncthreads();
    float* output = reinterpret_cast<float*>(smem.work);
#pragma unroll
    for (int qt = 0; qt < QueryTiles; ++qt)
      wmma::store_matrix_sync(output + qt * 16 * 128 + warp * 16, state[qt],
                              128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < QueryRows * 128; i += 256) {
      const int row = i / 128;
      if (query_begin + row < end)
        out[((query_begin + row) * 24 + head) * 128 + i % 128] =
            __float2half_rn(output[i]);
    }
    __syncthreads();
  }
}

template <int QueryRows>
int launch(const void* q, const void* k, const void* v, const void* h,
           const float* g, const void* cu_seqlens, const void* chunk_indices,
           void* out, Gdn75Shape shape, cudaStream_t stream) {
  const dim3 grid(shape.chunks * (64 / QueryRows), 8);
  output<QueryRows><<<grid, 256, 0, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(k),
      static_cast<const half*>(v), static_cast<const half*>(h), g, cu_seqlens,
      chunk_indices, static_cast<half*>(out), shape);
  return static_cast<int>(cudaGetLastError());
}
}  // namespace gdn75
