// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
namespace glm_mixed_gemv_v4 {
__device__ __forceinline__ int4 load_weight8(
    const unsigned char* __restrict__ payload, unsigned long long group_offset,
    int width, int n, int k, int N, int K) {
  const int first_n = n & ~63;
  const int rows = min(64, N - first_n);
  const size_t before = size_t(first_n & 255) * K + size_t(k & ~31) * rows;
  const auto tile = payload + group_offset + before * width / 8;
  if (width == 16)
    return *reinterpret_cast<const int4*>(tile +
                                          ((n & 63) * 32 + (k & 31)) * 2);
  const auto base = tile + (n & 63) * 16 + (k & 31) / 2;
  const unsigned a = *reinterpret_cast<const unsigned*>(base);
  const unsigned b = *reinterpret_cast<const unsigned*>(base + rows * 16);
  const unsigned c = *reinterpret_cast<const unsigned*>(base + rows * 32);
  int4 result;
  auto* dst = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    unsigned source = pair % 2 ? b : a;
    unsigned encoded;
    if (pair < 2)
      encoded = source & 0xbff8bff8u;
    else {
      unsigned low = __byte_perm(c, 0u, pair == 2 ? 0x4140 : 0x4342) << 3;
      encoded =
          low | ((source & 0x00070007u) << 11) | ((source & 0x40004000u) << 1);
    }
    unsigned decoded;
    asm("mul.rn.f16x2 %0, %1, %2;"
        : "=r"(decoded)
        : "r"(encoded), "r"(0x30003000u));
    dst[pair] = decoded;
  }
  return result;
}
}  // namespace glm_mixed_gemv_v4
