// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME rtx8000_marlin_safety_guarded
#include "../kernel.h"
#include "guarded_marlin_template.cuh"
#include "launch_impl.cuh"

MarlinSafetyResources marlin_safety_resources(int block_m, int device) {
  return rtx8000_marlin_safety_guarded::resources(block_m, device);
}
void launch_marlin_safety(const MarlinSafetyArgs& args) {
  rtx8000_marlin_safety_guarded::launch(args);
}
