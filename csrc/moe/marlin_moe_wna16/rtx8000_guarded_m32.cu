// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME rtx8000_marlin_guarded
#include "rtx8000_guarded.h"
#include "guarded_marlin_template.cuh"

namespace rtx8000_marlin_guarded {
GuardedMarlinFuncPtr get_m32_kernel() {
  return Marlin<vllm::kFloat16.id(), vllm::kU8B128.id(), vllm::kFloat16.id(),
                vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, -1, false>;
}
}  // namespace rtx8000_marlin_guarded
