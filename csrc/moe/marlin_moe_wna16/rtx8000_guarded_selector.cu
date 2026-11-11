// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME rtx8000_marlin_guarded
#include "rtx8000_guarded.h"

namespace rtx8000_marlin_guarded {
GuardedMarlinFuncPtr get_kernel(int thread_m_blocks) {
  switch (thread_m_blocks) {
    case 1:
      return get_m16_kernel();
    case 2:
      return get_m32_kernel();
    default:
      return nullptr;
  }
}
}  // namespace rtx8000_marlin_guarded
