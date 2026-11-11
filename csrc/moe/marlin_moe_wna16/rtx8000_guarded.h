// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel.h"

namespace rtx8000_marlin_guarded {
using GuardedMarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

GuardedMarlinFuncPtr get_m16_kernel();
GuardedMarlinFuncPtr get_m32_kernel();
GuardedMarlinFuncPtr get_kernel(int thread_m_blocks);
}  // namespace rtx8000_marlin_guarded
