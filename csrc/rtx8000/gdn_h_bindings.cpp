// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>

#include "gdn_h_launch.h"

namespace {
void gdn_h75(const at::Tensor& k, const at::Tensor& w, const at::Tensor& u,
             const at::Tensor& g, const at::Tensor& h0,
             const at::Tensor& cu_seqlens, const at::Tensor& chunk_offsets,
             at::Tensor h, at::Tensor v_new, at::Tensor ht, int64_t variant) {
  TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kHalf && k.dim() == 4 &&
                  k.size(0) == 1 && k.size(2) == 8 && k.size(3) == 128,
              "SM75 GDN state requires FP16 K [1,T,8,128]");
  for (const auto& t :
       {k, w, u, g, h0, cu_seqlens, chunk_offsets, h, v_new, ht})
    TORCH_CHECK(t.is_cuda() && t.device() == k.device() && t.is_contiguous(),
                "SM75 GDN state tensors must be contiguous on one CUDA device");
  TORCH_CHECK(u.scalar_type() == at::kHalf && u.dim() == 4 && u.size(0) == 1 &&
                  u.size(1) == k.size(1) && u.size(2) == 24 &&
                  u.size(3) == 128 && w.scalar_type() == at::kHalf &&
                  w.sizes() == u.sizes() && v_new.scalar_type() == at::kHalf &&
                  v_new.sizes() == u.sizes(),
              "SM75 GDN state requires FP16 W/U/residual [1,T,24,128]");
  TORCH_CHECK(g.scalar_type() == at::kFloat && g.dim() == 3 && g.size(0) == 1 &&
                  g.size(1) == k.size(1) && g.size(2) == 24,
              "SM75 GDN state requires FP32 gates [1,T,24]");
  TORCH_CHECK(
      (cu_seqlens.scalar_type() == at::kInt ||
       cu_seqlens.scalar_type() == at::kLong) &&
          cu_seqlens.dim() == 1 && cu_seqlens.size(0) >= 2 &&
          cu_seqlens.size(0) - 1 <= std::numeric_limits<int>::max() / 24 &&
          (chunk_offsets.scalar_type() == at::kInt ||
           chunk_offsets.scalar_type() == at::kLong) &&
          chunk_offsets.sizes() == cu_seqlens.sizes(),
      "SM75 GDN state requires INT32/INT64 sequence/chunk offsets");
  TORCH_CHECK(
      h0.scalar_type() == at::kFloat && h0.dim() == 4 &&
          h0.size(0) == cu_seqlens.size(0) - 1 && h0.size(1) == 24 &&
          h0.size(2) == 128 && h0.size(3) == 128 &&
          ht.scalar_type() == at::kFloat && ht.sizes() == h0.sizes(),
      "SM75 GDN initial/final state must be FP32 [sequences,24,128,128]");
  TORCH_CHECK(h.scalar_type() == at::kHalf && h.dim() == 5 && h.size(0) == 1 &&
                  h.size(2) == 24 && h.size(3) == 128 && h.size(4) == 128 &&
                  (k.size(1) == 0 || h.size(1) > 0),
              "SM75 GDN snapshots must be FP16 [1,chunks,24,128,128]");
  TORCH_CHECK(variant == 0 || variant == 1,
              "SM75 GDN state variant must be 0 or 1");
  for (const auto& t : {k, w})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "SM75 GDN K/W must be 16-byte aligned");
  for (const auto& output : {h, v_new, ht})
    for (const auto& input : {k, w, u, g, h0, cu_seqlens, chunk_offsets})
      at::assert_no_overlap(output, input);
  at::assert_no_overlap(h, v_new);
  at::assert_no_overlap(h, ht);
  at::assert_no_overlap(v_new, ht);
  c10::cuda::CUDAGuard guard(k.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5,
              "SM75 GDN state requires Turing");
  GdnH75Shape shape{k.size(1), h.size(1),
                    static_cast<int>(cu_seqlens.size(0) - 1),
                    cu_seqlens.scalar_type() == at::kLong,
                    chunk_offsets.scalar_type() == at::kLong};
  GdnH75Launch launch = variant == 0 ? gdn_h75_v8_launch : gdn_h75_v16_launch;
  C10_CUDA_CHECK(static_cast<cudaError_t>(
      launch(k.data_ptr(), w.data_ptr(), u.data_ptr(), g.data_ptr<float>(),
             h0.data_ptr<float>(), cu_seqlens.data_ptr(),
             chunk_offsets.data_ptr(), h.data_ptr(), v_new.data_ptr(),
             ht.data_ptr<float>(), shape, at::cuda::getCurrentCUDAStream())));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_rtx8000_C, m) {
  m.def(
      "gdn_h(Tensor k, Tensor w, Tensor u, Tensor g, Tensor h0, "
      "Tensor cu_seqlens, Tensor chunk_offsets, Tensor(a!) h, "
      "Tensor(b!) v_new, Tensor(c!) final_state, int variant=0) -> ()");
}

TORCH_LIBRARY_IMPL(_rtx8000_C, CUDA, m) { m.impl("gdn_h", &gdn_h75); }
