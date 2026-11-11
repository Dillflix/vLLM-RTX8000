// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <algorithm>
#include "launch.h"

namespace {
at::Tensor gate_up(const at::Tensor& x, at::Tensor output,
                   const at::Tensor& packed, const at::Tensor& scales,
                   at::Tensor workspace, const at::Tensor& sorted,
                   const at::Tensor& experts, const at::Tensor& padded,
                   bool fused) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.size(1) == 2560 &&
                  (x.size(0) == 1 || x.size(0) == 1600 || x.size(0) == 2048),
              "Prototype requires FP16 [1|1600|2048,2560]");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5, "Prototype requires SM75");
  for (const auto& t :
       {x, output, packed, scales, workspace, sorted, experts, padded}) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous(),
                "Prototype tensors must be contiguous on one CUDA device");
  }
  TORCH_CHECK(packed.scalar_type() == at::kInt && packed.dim() == 3 &&
                  packed.size(0) == 512 && packed.size(1) == 160 &&
                  packed.size(2) == 2560,
              "Expected packed512-expert per-channel INT8 gate/up weights");
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 3 &&
                  scales.size(0) == 512 && scales.size(1) == 1 &&
                  scales.size(2) == 640,
              "Expected FP16 per-channel scales");
  TORCH_CHECK(output.scalar_type() == at::kHalf && output.dim() == 2 &&
                  output.size(0) == x.size(0) * 10 &&
                  output.size(1) == (fused ? 320 : 640),
              "Incorrect prototype output shape");
  for (const auto& t : {workspace, sorted, experts, padded}) {
    TORCH_CHECK(t.scalar_type() == at::kInt && t.dim() == 1,
                "Routing/lock tensors must be flat INT32");
  }
  const int block_m = x.size(0) == 1 ? 16 : 32;
  TORCH_CHECK(workspace.numel() >= prop->multiProcessorCount * 4 &&
                  sorted.numel() >= x.size(0) * 10 && padded.numel() == 1 &&
                  experts.numel() * block_m >= sorted.numel(),
              "Insufficient routing or lock workspace");
  const int64_t partial_size = std::min<int64_t>(
      640 * sorted.numel(), prop->multiProcessorCount * 4L * block_m * 256);
  auto partial = at::empty({partial_size}, x.options().dtype(at::kFloat));
  MarlinFusionArgs args{x.data_ptr(),
                        packed.data_ptr(),
                        scales.data_ptr(),
                        output.data_ptr(),
                        partial.data_ptr(),
                        workspace.data_ptr<int>(),
                        sorted.data_ptr<int>(),
                        experts.data_ptr<int>(),
                        padded.data_ptr<int>(),
                        static_cast<int>(x.size(0)),
                        block_m,
                        x.get_device(),
                        prop->multiProcessorCount,
                        at::cuda::getCurrentCUDAStream(x.get_device())};
  if (fused)
    launch_marlin_fusion(args);
  else
    launch_marlin_control(args);
  return output;
}

std::vector<int64_t> resources(int64_t block_m, int64_t device, bool fused) {
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, device));
  auto info = fused ? marlin_fusion_resources(block_m, device)
                    : marlin_control_resources(block_m, device);
  return {info.registers, info.local_bytes, info.shared_bytes,
          info.active_ctas};
}
}  // namespace

TORCH_LIBRARY(_rtx8000_marlin_fusion, m) {
  m.def(
      "gate_up(Tensor x, Tensor(a!) output, Tensor packed, Tensor scales, "
      "Tensor(b!) workspace, Tensor sorted, Tensor experts, Tensor padded, "
      "bool fused) -> Tensor(a!)");
  m.def("resources(int block_m, int device, bool fused) -> int[]", &resources);
}
TORCH_LIBRARY_IMPL(_rtx8000_marlin_fusion, CUDA, m) {
  m.impl("gate_up", &gate_up);
}
