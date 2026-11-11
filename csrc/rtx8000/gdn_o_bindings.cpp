// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "gdn_o_launch.h"

namespace {
void gdn_o75(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
             const at::Tensor& h, const at::Tensor& g,
             const at::Tensor& cu_seqlens, const at::Tensor& chunk_indices,
             at::Tensor out, double scale, int64_t variant) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 4 &&
                  q.size(0) == 1 && q.size(2) == 8 && q.size(3) == 128,
              "SM75 GDN output requires FP16 Q [1,T,8,128]");
  for (const auto& t : {q, k, v, h, g, cu_seqlens, chunk_indices, out}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous(),
                "SM75 GDN tensors must be contiguous on the same CUDA device");
  }
  TORCH_CHECK(k.scalar_type() == at::kHalf && k.sizes() == q.sizes() &&
                  v.scalar_type() == at::kHalf && v.dim() == 4 &&
                  v.size(0) == 1 && v.size(1) == q.size(1) && v.size(2) == 24 &&
                  v.size(3) == 128 && out.scalar_type() == at::kHalf &&
                  out.sizes() == v.sizes(),
              "SM75 GDN requires matching FP16 K and V/output [1,T,24,128]");
  TORCH_CHECK(g.scalar_type() == at::kFloat && g.dim() == 3 && g.size(0) == 1 &&
                  g.size(1) == q.size(1) && g.size(2) == 24,
              "SM75 GDN requires FP32 gates [1,T,24]");
  TORCH_CHECK(
      (cu_seqlens.scalar_type() == at::kInt ||
       cu_seqlens.scalar_type() == at::kLong) &&
          cu_seqlens.dim() == 1 && cu_seqlens.size(0) >= 2 &&
          cu_seqlens.size(0) <= std::numeric_limits<int>::max() &&
          chunk_indices.scalar_type() == cu_seqlens.scalar_type() &&
          chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
          chunk_indices.size(0) <= std::numeric_limits<int>::max() / 4,
      "SM75 GDN requires same-type INT32/INT64 sequence/chunk metadata");
  TORCH_CHECK(h.scalar_type() == at::kHalf && h.dim() == 5 && h.size(0) == 1 &&
                  h.size(1) == chunk_indices.size(0) && h.size(2) == 24 &&
                  h.size(3) == 128 && h.size(4) == 128,
              "SM75 GDN requires FP16 H [1,chunks,24,128,128] in V,K order");
  TORCH_CHECK(std::isfinite(scale) &&
                  std::isfinite(static_cast<float>(scale)) &&
                  (variant == 0 || variant == 1),
              "SM75 GDN scale must be finite and variant must be 0 or 1");
  for (const auto& t : {q, k, v, h}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 32 == 0,
                "SM75 GDN inputs must be 32-byte aligned");
  }
  for (const auto& t : {q, k, v, h, g, cu_seqlens, chunk_indices})
    at::assert_no_overlap(out, t);
  c10::cuda::CUDAGuard guard(q.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5,
              "SM75 GDN output requires Turing");
  TORCH_CHECK(q.size(1) == 0 || chunk_indices.size(0) > 0,
              "SM75 GDN requires chunk metadata for nonempty input");
  if (q.size(1) == 0) return;
  Gdn75Shape shape{q.size(1), static_cast<int>(chunk_indices.size(0)),
                   static_cast<int>(cu_seqlens.size(0) - 1),
                   cu_seqlens.scalar_type() == at::kLong,
                   static_cast<float>(scale)};
  auto launch = variant == 0 ? gdn_o75_q16_launch : gdn_o75_q32_launch;
  C10_CUDA_CHECK(static_cast<cudaError_t>(launch(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), h.data_ptr(),
      g.data_ptr<float>(), cu_seqlens.data_ptr(), chunk_indices.data_ptr(),
      out.data_ptr(), shape, at::cuda::getCurrentCUDAStream())));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_rtx8000_C, m) {
  m.def(
      "gdn_o(Tensor q, Tensor k, Tensor v, Tensor h, Tensor g, "
      "Tensor cu_seqlens, Tensor chunk_indices, Tensor(a!) out, "
      "float scale, int variant=0) -> ()");
}

TORCH_LIBRARY_IMPL(_rtx8000_C, CUDA, m) { m.impl("gdn_o", &gdn_o75); }
