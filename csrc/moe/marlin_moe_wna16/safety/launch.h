// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>

struct MarlinSafetyArgs {
  const void* x;
  const void* packed;
  const void* scales;
  void* output;
  void* partial;
  int* locks;
  const int* sorted;
  const int* experts;
  const int* padded;
  const float* topk_weights;
  int tokens;
  bool down;
  int block_m;
  int device;
  int sms;
  cudaStream_t stream;
};

struct MarlinSafetyResources {
  int registers;
  int local_bytes;
  int shared_bytes;
  int active_ctas;
};

MarlinSafetyResources marlin_safety_resources(int block_m, int device);
MarlinSafetyResources marlin_safety_control_resources(int block_m, int device);
void launch_marlin_safety(const MarlinSafetyArgs& args);
void launch_marlin_safety_control(const MarlinSafetyArgs& args);
