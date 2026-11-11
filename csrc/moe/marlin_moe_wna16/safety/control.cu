// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME rtx8000_marlin_safety_control
#include "../kernel.h"
#include "../marlin_template.h"
#include "launch_impl.cuh"

MarlinSafetyResources marlin_safety_control_resources(int block_m, int device) {
  return rtx8000_marlin_safety_control::resources(block_m, device);
}
void launch_marlin_safety_control(const MarlinSafetyArgs& args) {
  rtx8000_marlin_safety_control::launch(args);
}
