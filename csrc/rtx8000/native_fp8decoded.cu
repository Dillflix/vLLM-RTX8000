// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "native_indexer.cuh"
using namespace glm_native_indexer;

extern "C" int glm_score_fp8decoded(const void* q, const void* k, const void* s,
                                    const void* w, const void* starts,
                                    const void* ends, void* out, int rows,
                                    int n, int width, void* stream) {
  score_debug<__half, true>
      <<<dim3(rows, (width + 63) / 64), threads, 0, (cudaStream_t)stream>>>(
          (const uint8_t*)q, (const __half*)k, (const float*)s, (const float*)w,
          (const int*)starts, (const int*)ends, (float*)out, n, width);
  return int(cudaGetLastError());
}

extern "C" int glm_local_fp8decoded(const void* q, const void* k, const void* s,
                                    const void* w, const void* starts,
                                    const void* ends, void* candidates,
                                    int rows, int n, int width, void* stream) {
  const int parts = (width + 4095) / 4096;
  fused_local<4096, false, __half, true>
      <<<dim3(rows, parts), threads, 0, (cudaStream_t)stream>>>(
          (const uint8_t*)q, (const __half*)k, (const float*)s, (const float*)w,
          (const int*)starts, (const int*)ends, (uint64_t*)candidates, n, width,
          parts);
  return int(cudaGetLastError());
}

extern "C" int glm_local_fp8decoded_runtime(const void* q, const void* k,
                                            const void* s, const void* w,
                                            const void* starts,
                                            const void* ends, void* candidates,
                                            int rows, int n, int width,
                                            void* stream) {
  const int parts = (width + 4095) / 4096;
  fused_local<4096, true, __half, true>
      <<<dim3(rows, parts), threads, 0, (cudaStream_t)stream>>>(
          (const uint8_t*)q, (const __half*)k, (const float*)s, (const float*)w,
          (const int*)starts, (const int*)ends, (uint64_t*)candidates, n, width,
          parts);
  return int(cudaGetLastError());
}

extern "C" int glm_resources_fp8decoded(int* out) {
  cudaFuncAttributes a;
  auto e = cudaFuncGetAttributes(&a, fused_local<4096, true, __half, true>);
  if (e != cudaSuccess) return int(e);
  out[0] = a.numRegs;
  out[1] = int(a.sharedSizeBytes);
  out[2] = int(a.localSizeBytes);
  return int(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      out + 3, fused_local<4096, true, __half, true>, threads, 0));
}
