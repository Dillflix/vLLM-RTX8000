// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cstdint>
#include "lossless_head_tile_launch.h"

namespace {
bool overlap(const at::Tensor& a, const at::Tensor& b) {
  if (!a.numel() || !b.numel()) return false;
  auto begin_a = reinterpret_cast<uintptr_t>(a.data_ptr());
  auto begin_b = reinterpret_cast<uintptr_t>(b.data_ptr());
  auto end_a = begin_a + a.numel() * a.element_size();
  auto end_b = begin_b + b.numel() * b.element_size();
  return begin_a < end_b && begin_b < end_a;
}

void lossless_head(const at::Tensor& x, const at::Tensor& packed,
                   at::Tensor out, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && packed.is_cuda() && out.is_cuda() &&
                  x.device() == packed.device() && x.device() == out.device(),
              "Lossless head tensors must share a CUDA device");
  TORCH_CHECK(x.dim() == 2 && packed.dim() == 2 && out.dim() == 2 &&
                  x.scalar_type() == at::kHalf &&
                  packed.scalar_type() == at::kByte &&
                  out.scalar_type() == at::kHalf && x.is_contiguous() &&
                  packed.is_contiguous() && out.is_contiguous(),
              "Lossless head requires contiguous FP16 activations/output and "
              "packed bytes");
  TORCH_CHECK(x.size(0) >= 0 && x.size(0) <= 15 && x.size(1) == 4096 &&
                  packed.size(0) == 77440 && packed.size(1) == 6144 &&
                  out.size(0) == x.size(0) && out.size(1) == 77440,
              "Lossless head supports GLM TP2 shape77440x4096 with0..15 rows");
  TORCH_CHECK(variant == 0,
              "Only the qualified N64 tile head variant0 is supported");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "Lossless head buffers must be16-byte aligned");
  TORCH_CHECK(!overlap(out, x) && !overlap(out, packed),
              "Lossless head output must not overlap either input");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5,
              "Lossless head requires SM75");
  // Do not launch a zero-sized grid or perform an unrelated collective.
  if (x.size(0) == 0) return;
  const int error = lossless_head_tile_launch(x.data_ptr(), packed.data_ptr(),
                                              out.data_ptr(), int(x.size(0)),
                                              at::cuda::getCurrentCUDAStream());
  C10_CUDA_CHECK(static_cast<cudaError_t>(error));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_rtx8000_C, m) {
  m.def(
      "lossless_head(Tensor x, Tensor packed, Tensor(a!) out, int variant) -> "
      "()");
}
TORCH_LIBRARY_IMPL(_rtx8000_C, CUDA, m) {
  m.impl("lossless_head", lossless_head);
}
