// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME rtx8000_marlin_control
#include "../kernel.h"
#include "../marlin_template.h"
#include "launch_impl.cuh"

MarlinFusionResources marlin_control_resources(int block_m, int device) {
  return rtx8000_marlin_control::resources(block_m, device);
}
void launch_marlin_control(const MarlinFusionArgs& args) {
  rtx8000_marlin_control::launch(args);
}
