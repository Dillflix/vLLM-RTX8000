// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <algorithm>
#include "launch.h"

namespace {
at::Tensor gemm(const at::Tensor& x, at::Tensor output,
                const at::Tensor& packed, const at::Tensor& scales,
                at::Tensor workspace, const at::Tensor& sorted,
                const at::Tensor& experts, const at::Tensor& padded,
                const at::Tensor& topk_weights, int64_t tokens, bool down,
                bool guarded) {
  TORCH_CHECK(tokens == 1 || tokens == 1600 || tokens == 2048,
              "Safety experiment requires M1/M1600/M2048");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2,
              "Safety experiment requires FP16 CUDA activations");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5, "Requires SM75");
  const int n = down ? 2560 : 640;
  const int k = down ? 320 : 2560;
  const int block_m = tokens == 1 ? 16 : 32;
  for (const auto& t : {x, output, packed, scales, workspace, sorted, experts,
                        padded, topk_weights}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous(),
                "Safety tensors must be contiguous on one CUDA device");
  }
  TORCH_CHECK(x.size(0) == tokens * (down ? 10 : 1) && x.size(1) == k,
              "Incorrect activation dimensions");
  TORCH_CHECK(output.scalar_type() == at::kHalf && output.dim() == 2 &&
                  output.size(0) == tokens * 10 && output.size(1) == n,
              "Incorrect output dimensions");
  TORCH_CHECK(packed.scalar_type() == at::kInt && packed.dim() == 3 &&
                  packed.size(0) == 512 && packed.size(1) == k / 16 &&
                  packed.size(2) == n * 4,
              "Expected packed per-channel INT8 expert weights");
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 3 &&
                  scales.size(0) == 512 && scales.size(1) == 1 &&
                  scales.size(2) == n,
              "Expected FP16 per-channel scales");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.dim() == 2 && topk_weights.size(0) == tokens &&
                  topk_weights.size(1) == 10,
              "Expected FP32 top-10 routing weights");
  for (const auto& t : {workspace, sorted, experts, padded}) {
    TORCH_CHECK(t.scalar_type() == at::kInt && t.dim() == 1,
                "Routing and lock tensors must be flat INT32");
  }
  TORCH_CHECK(workspace.numel() >= prop->multiProcessorCount * 4 &&
                  sorted.numel() >= tokens * 10 && padded.numel() == 1 &&
                  experts.numel() * block_m >= sorted.numel(),
              "Insufficient routing or lock workspace");
  const int64_t partial_size = std::min<int64_t>(
      n * sorted.numel(), prop->multiProcessorCount * 4L * block_m * 256);
  auto partial = at::empty({partial_size}, x.options().dtype(at::kFloat));
  MarlinSafetyArgs args{x.data_ptr(),
                        packed.data_ptr(),
                        scales.data_ptr(),
                        output.data_ptr(),
                        partial.data_ptr(),
                        workspace.data_ptr<int>(),
                        sorted.data_ptr<int>(),
                        experts.data_ptr<int>(),
                        padded.data_ptr<int>(),
                        topk_weights.data_ptr<float>(),
                        static_cast<int>(tokens),
                        down,
                        block_m,
                        x.get_device(),
                        prop->multiProcessorCount,
                        at::cuda::getCurrentCUDAStream(x.get_device())};
  if (guarded)
    launch_marlin_safety(args);
  else
    launch_marlin_safety_control(args);
  return output;
}

std::vector<int64_t> resources(int64_t block_m, int64_t device, bool guarded) {
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, device));
  auto info = guarded ? marlin_safety_resources(block_m, device)
                      : marlin_safety_control_resources(block_m, device);
  return {info.registers, info.local_bytes, info.shared_bytes,
          info.active_ctas};
}
}  // namespace

TORCH_LIBRARY(_rtx8000_marlin_safety, m) {
  m.def(
      "gemm(Tensor x, Tensor(a!) output, Tensor packed, Tensor scales, "
      "Tensor(b!) workspace, Tensor sorted, Tensor experts, Tensor padded, "
      "Tensor topk_weights, int tokens, bool down, bool guarded) -> "
      "Tensor(a!)");
  m.def("resources(int block_m, int device, bool guarded) -> int[]",
        &resources);
}
TORCH_LIBRARY_IMPL(_rtx8000_marlin_safety, CUDA, m) { m.impl("gemm", &gemm); }
