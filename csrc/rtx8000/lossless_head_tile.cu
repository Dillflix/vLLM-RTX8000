// SPDX-License-Identifier: Apache-2.0
#include "lossless_head_tile_kernel.cuh"
#include "lossless_head_tile_launch.h"
int lossless_head_tile_launch(const void* x, const void* packed, void* out,
                              int rows, void* stream) {
  return head_mma75_12_tileplanes_v1::launch<64, 1, true>(x, packed, out, rows,
                                                          77440, 4096, stream);
}
