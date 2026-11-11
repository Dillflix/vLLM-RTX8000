// SPDX-License-Identifier: Apache-2.0
#include "launch.h"
#include <cuda_runtime.h>

#define RTX_QSA_VARIANT(ID, NAME, KEYS, WARPS, PREFETCH)                       \
  extern "C" int sparse75_##NAME(const void*, const void*, const void*,        \
                                 const void*, const void*, const void*, void*, \
                                 void*, void*, Qsa75Shape, void*);             \
  extern "C" int sparse75_##NAME##_resources(int*, int*, int*);
#include "qsa_variants.inc"
#undef RTX_QSA_VARIANT
extern "C" void qsa75_resources(int*, int*, int*);

extern "C" int sparse75_variant_launch(const void* q, const void* kv,
                                       const void* idx, const void* table,
                                       const void* req, const void* gate,
                                       void* out, void* partial, void* lse,
                                       Qsa75Shape shape, void* stream,
                                       int variant) {
  switch (variant) {
    case 0:
      return sparse75_launch(q, kv, idx, table, req, gate, out, partial, lse,
                             shape, stream);
#define RTX_QSA_VARIANT(ID, NAME, KEYS, WARPS, PREFETCH)                    \
  case ID:                                                                  \
    return sparse75_##NAME(q, kv, idx, table, req, gate, out, partial, lse, \
                           shape, stream);
#include "qsa_variants.inc"
#undef RTX_QSA_VARIANT
    default:
      return int(cudaErrorInvalidValue);
  }
}

extern "C" int qsa75_variant_resources(int variant, int* regs, int* shared,
                                       int* blocks) {
  switch (variant) {
    case 0:
      qsa75_resources(regs, shared, blocks);
      return 0;
#define RTX_QSA_VARIANT(ID, NAME, KEYS, WARPS, PREFETCH) \
  case ID:                                               \
    return sparse75_##NAME##_resources(regs, shared, blocks);
#include "qsa_variants.inc"
#undef RTX_QSA_VARIANT
    default:
      return int(cudaErrorInvalidValue);
  }
}
