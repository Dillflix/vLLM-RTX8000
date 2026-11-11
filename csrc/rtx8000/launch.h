// SPDX-License-Identifier: Apache-2.0
#pragma once
struct Qsa75Shape {
  int rows, pages, page_size, table_width, table_stride, requests, splits;
  int topk, index_stride, query_stride, output_stride, gate_stride;
  long long cache_block_stride;
};
extern "C" int sparse75_launch(const void*, const void*, const void*,
                               const void*, const void*, const void*, void*,
                               void*, void*, Qsa75Shape, void*);
extern "C" int dense75_launch(const void*, const void*, void*, int, int, int,
                              int, void*);
extern "C" int sparse75_variant_launch(const void*, const void*, const void*,
                                       const void*, const void*, const void*,
                                       void*, void*, void*, Qsa75Shape, void*,
                                       int);
