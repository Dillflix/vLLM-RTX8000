// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>

struct MarlinFusionArgs {
  const void* x;
  const void* packed;
  const void* scales;
  void* output;
  void* partial;
  int* locks;
  const int* sorted;
  const int* experts;
  const int* padded;
  int tokens;
  int block_m;
  int device;
  int sms;
  cudaStream_t stream;
};

struct MarlinFusionResources {
  int registers;
  int local_bytes;
  int shared_bytes;
  int active_ctas;
};

MarlinFusionResources marlin_fusion_resources(int block_m, int device);
MarlinFusionResources marlin_control_resources(int block_m, int device);
void launch_marlin_fusion(const MarlinFusionArgs& args);
void launch_marlin_control(const MarlinFusionArgs& args);
