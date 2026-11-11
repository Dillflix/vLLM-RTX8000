// SPDX-License-Identifier: Apache-2.0
#include "launch.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
using namespace nvcuda;

// Specialized kernel for the measured TP2 layout (12 query heads, one KV
// head, D256, packed FP16 KV). No approximation of the selected attention set.
// One CTA shares Q/K/V across all heads; 8 warps distribute the D256
// accumulator.
__global__ void qsa75(const half* q, const half* packed, const int* indices,
                      const int* table, const int* requests, const half* gate,
                      half* output, float* partial, float* lse,
                      Qsa75Shape shape) {
  __shared__ __align__(32) half sq[16 * 256];
  __shared__ __align__(32) half kv[32 * 256];
  __shared__ __align__(32) float score[16 * 32];
  __shared__ __align__(32) half prob[16 * 32];
  __shared__ __align__(32) float alpha_matrix[16 * 16];
  __shared__ float maximum[16], denominator[16];
  __shared__ long long offsets[32];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int M = shape.rows, pages = shape.pages, width = shape.table_width;
  const int nreq = shape.requests, splits = shape.splits;
  const int row = blockIdx.x, split = blockIdx.y;
  const int req = requests[row];
  const int count =
      min(shape.topk, max(0, indices[row * shape.index_stride + shape.topk]));
  for (int i = tid; i < 16 * 256; i += 256)
    sq[i] = i / 256 < 12 ? q[row * shape.query_stride + i] : __float2half(0.f);
  if (tid < 16) {
    maximum[tid] = -CUDART_INF_F;
    denominator[tid] = 0.f;
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
  wmma::fill_fragment(acc[0], 0.f);
  wmma::fill_fragment(acc[1], 0.f);
  __syncthreads();
  for (int tile = split; tile < (count + 31) / 32; tile += splits) {
    if (tid < 32) {
      int column = tile * 32 + tid;
      int logical =
          column < count ? indices[row * shape.index_stride + column] : -1;
      int pg = logical >= 0 ? logical / shape.page_size : -1;
      int physical = (req >= 0 && req < nreq && pg >= 0 && pg < width)
                         ? table[req * shape.table_stride + pg]
                         : -1;
      offsets[tid] = (physical >= 0 && physical < pages)
                         ? (long long)physical * shape.cache_block_stride +
                               (logical % shape.page_size) * 512
                         : -1;
    }
    __syncthreads();
    for (int vec = tid; vec < 32 * 32; vec += 256) {
      long long off = offsets[vec / 32];
      int4 value = {0, 0, 0, 0};
      if (off >= 0)
        value = *reinterpret_cast<const int4*>(packed + off + (vec % 32) * 8);
      reinterpret_cast<int4*>(kv)[vec] = value;
    }
    __syncthreads();
    if (warp < 2) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
      wmma::fill_fragment(s, 0.f);
      for (int k = 0; k < 256; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, sq + k, 256);
        wmma::load_matrix_sync(b, kv + warp * 16 * 256 + k, 256);
        wmma::mma_sync(s, a, b, s);
      }
      wmma::store_matrix_sync(score + warp * 16, s, 32, wmma::mem_row_major);
    }
    __syncthreads();
#pragma unroll
    for (int h = warp; h < 16; h += 8) {
      float s = offsets[lane] >= 0
                    ? score[h * 32 + lane] * (1.4426950408889634f / 16.f)
                    : -CUDART_INF_F;
      float mx = s;
#pragma unroll
      for (int d = 16; d; d /= 2)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
      float next = fmaxf(mx, maximum[h]);
      float alpha = isfinite(next) ? exp2f(maximum[h] - next) : 0.f;
      float p = isfinite(s) ? exp2f(s - next) : 0.f;
      prob[h * 32 + lane] = __float2half(p);
      float sum = p;
#pragma unroll
      for (int d = 16; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
      if (lane < 16) alpha_matrix[h * 16 + lane] = alpha;
      // Every lane must finish reading the old shared maximum before lane 0
      // overwrites it. Shuffle synchronization does not order shared memory.
      __syncwarp();
      if (!lane) {
        maximum[h] = next;
        denominator[h] = denominator[h] * alpha + sum;
      }
    }
    // K is dead. Reuse its shared storage for V before the second tensor
    // product.
    for (int vec = tid; vec < 32 * 32; vec += 256) {
      long long off = offsets[vec / 32];
      int4 value = {0, 0, 0, 0};
      if (off >= 0)
        value =
            *reinterpret_cast<const int4*>(packed + off + 256 + (vec % 32) * 8);
      reinterpret_cast<int4*>(kv)[vec] = value;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> alpha;
    wmma::load_matrix_sync(alpha, alpha_matrix, 16, wmma::mem_row_major);
#pragma unroll
    for (int z = 0; z < 2; ++z) {
#pragma unroll
      for (int i = 0; i < acc[z].num_elements; ++i) acc[z].x[i] *= alpha.x[i];
#pragma unroll
      for (int k = 0; k < 32; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
        wmma::load_matrix_sync(a, prob + k, 32);
        wmma::load_matrix_sync(b, kv + k * 256 + (warp + z * 8) * 16, 256);
        wmma::mma_sync(acc[z], a, b, acc[z]);
      }
    }
    __syncthreads();
  }
  float* result = reinterpret_cast<float*>(kv);
#pragma unroll
  for (int z = 0; z < 2; ++z)
    wmma::store_matrix_sync(result + (warp + z * 8) * 16, acc[z], 256,
                            wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < 12 * 256; i += 256) {
    float value =
        denominator[i / 256] > 0 ? result[i] / denominator[i / 256] : 0.f;
    if (splits == 1) {
      value = __half2float(__float2half(value));
      output[row * shape.output_stride + i] = __float2half(
          value /
          (1.f + expf(-__half2float(gate[row * shape.gate_stride + i]))));
    } else
      partial[((long long)split * M + row) * 3072 + i] = value;
  }
  if (splits > 1 && tid < 12)
    lse[(split * M + row) * 12 + tid] =
        denominator[tid] > 0 ? maximum[tid] + log2f(denominator[tid])
                             : -CUDART_INF_F;
}
extern "C" int sparse75_launch(const void* q, const void* kv, const void* idx,
                               const void* table, const void* req,
                               const void* gate, void* out, void* partial,
                               void* lse, Qsa75Shape shape, void* stream) {
  qsa75<<<dim3(shape.rows, shape.splits), 256, 0,
          static_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(q), static_cast<const half*>(kv),
      static_cast<const int*>(idx), static_cast<const int*>(table),
      static_cast<const int*>(req), static_cast<const half*>(gate),
      static_cast<half*>(out), static_cast<float*>(partial),
      static_cast<float*>(lse), shape);
  return int(cudaGetLastError());
}
extern "C" void qsa75_resources(int* regs, int* shared, int* blocks) {
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, qsa75);
  *regs = a.numRegs;
  *shared = a.sharedSizeBytes;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, qsa75, 256, 0);
}
