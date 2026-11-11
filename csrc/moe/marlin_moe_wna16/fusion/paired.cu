// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME rtx8000_marlin_paired
#include "../kernel.h"
#include "paired_marlin_template.cuh"
#include "launch_impl.cuh"

MarlinFusionResources marlin_fusion_resources(int block_m, int device) {
  return rtx8000_marlin_paired::resources(block_m, device);
}
void launch_marlin_fusion(const MarlinFusionArgs& args) {
  rtx8000_marlin_paired::launch(args);
}
