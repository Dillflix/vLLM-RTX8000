// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gemm.cuh"

namespace glm_lossless_cutlass_v26 {

// Diagnostic only: exercises the exact B iterator independently of MMA.
template <int BM, int BN>
__global__ void decode_iterator(uint8_t const* payload,
                                uint64_t const* metadata, uint16_t* output,
                                int n, int k, bool clear) {
  using Iterator = typename Types<BM, BN, true>::Mma::IteratorB;
  using Map = typename Iterator::ThreadMap;
  typename Iterator::Params params;
  params.metadata = metadata;
  int first_k = int(blockIdx.x) * 32;
  int first_n = int(blockIdx.y) * BN;
  Iterator iterator(
      params, reinterpret_cast<cutlass::half_t*>(const_cast<uint8_t*>(payload)),
      {k, n}, int(threadIdx.x), {first_k, first_n});
  typename Iterator::Fragment fragment;
  iterator.clear_mask(clear);
  iterator.load(fragment);
  auto initial = Map::initial_offset(int(threadIdx.x));
  auto const* words = reinterpret_cast<uint16_t const*>(&fragment);
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < Map::Iterations::kStrided; ++s) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < Map::Iterations::kContiguous; ++c) {
      int nn = first_n + initial.strided() + s * Map::Delta::kStrided;
      int kk = first_k + initial.contiguous() + c * Map::Delta::kContiguous;
      int index = 8 * (c + s * Map::Iterations::kContiguous);
      if (nn < n && kk + 7 < k) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i)
          output[uint64_t(nn) * k + kk + i] = words[index + i];
      }
    }
  }
}

template <int BM, int BN>
int launch_decode(void const* payload, void const* metadata, void* out, int n,
                  int k, int clear, void* stream) {
  if (!payload || !metadata || !out || n <= 0 || n % 8 || k <= 0 || k % 64 ||
      reinterpret_cast<uintptr_t>(payload) % 16)
    return -1;
  decode_iterator<BM, BN><<<dim3(k / 32, (n + BN - 1) / BN), 256, 0,
                            static_cast<cudaStream_t>(stream)>>>(
      static_cast<uint8_t const*>(payload),
      static_cast<uint64_t const*>(metadata), static_cast<uint16_t*>(out), n, k,
      clear != 0);
  return int(cudaGetLastError());
}

}  // namespace glm_lossless_cutlass_v26
