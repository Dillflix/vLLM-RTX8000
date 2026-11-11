// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <string>
#include <vector>
#include "launch.h"

namespace MARLIN_NAMESPACE_NAME {
using Kernel = void (*)(MARLIN_KERNEL_PARAMS);

Kernel selected_kernel(int block_m) {
  if (block_m == 16) {
    return Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(),
                  vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, -1, false>;
  }
  if (block_m == 32) {
    return Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(),
                  vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
  }
  throw std::invalid_argument("Marlin safety only instantiates M16 and M32");
}

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

MarlinSafetyResources resources(int block_m, int device) {
  struct Entry {
    int m, device;
    MarlinSafetyResources resources;
  };
  static thread_local std::vector<Entry> cache;
  for (const auto& entry : cache) {
    if (entry.m == block_m && entry.device == device) return entry.resources;
  }
  auto kernel = selected_kernel(block_m);
  const int shared = block_m == 16 ? 21248 : 25600;
  check_cuda(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  cudaFuncAttributes attributes;
  check_cuda(cudaFuncGetAttributes(&attributes, kernel));
  int active;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 128,
                                                           shared));
  MarlinSafetyResources result{attributes.numRegs,
                               static_cast<int>(attributes.localSizeBytes),
                               shared, active};
  cache.push_back({block_m, device, result});
  return result;
}

void launch(const MarlinSafetyArgs& a) {
  const auto info = resources(a.block_m, a.device);
  const int ctas = a.down && a.tokens == 1 ? 3 : 2;
  if (info.active_ctas < ctas) {
    throw std::runtime_error(
        "Safety variant cannot sustain the baseline CTA count");
  }
  auto kernel = selected_kernel(a.block_m);
  kernel<<<a.sms * ctas, 128, info.shared_bytes, a.stream>>>(
      static_cast<const int4*>(a.x), static_cast<const int4*>(a.packed),
      static_cast<int4*>(a.output), static_cast<int4*>(a.partial), nullptr,
      static_cast<const int4*>(a.scales), nullptr, nullptr, a.sorted, a.experts,
      a.padded, a.topk_weights, a.down ? 1 : 10, a.down,
      a.down ? a.tokens * 10 : a.tokens, a.down ? 2560 : 640,
      a.down ? 320 : 2560, a.locks, false, false, true);
  check_cuda(cudaGetLastError());
}
}  // namespace MARLIN_NAMESPACE_NAME
