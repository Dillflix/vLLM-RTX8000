// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <array>
#include <mutex>

#define PREFILL(NAME)                                                         \
  extern "C" int NAME(const void*, const void*, const void*, void*, int, int, \
                      int, void*);                                            \
  extern "C" int NAME##_configure(int*);
PREFILL(v26_m256n128_packed)
PREFILL(v26_m128n256_packed)
PREFILL(v29_m128n256_packed)
PREFILL(v30_m32n128_packed)
PREFILL(v30_m64n64_packed)
PREFILL(v30_m64n128_packed)
PREFILL(v30_m128n128_packed)
#undef PREFILL
extern "C" int m16n64_packed(const void*, const void*, const void*, void*,
                             void*, int, int, int, int, void*);
extern "C" int m16n64_packed_configure(int*);
extern "C" int glm_mixed_gemv_m1(const void*, const void*, const void*, void*,
                                 int, int, int, void*);

namespace {
using Prefill = int (*)(const void*, const void*, const void*, void*, int, int,
                        int, void*);
std::array<Prefill, 7> prefill = {v26_m256n128_packed, v26_m128n256_packed,
                                  v29_m128n256_packed, v30_m32n128_packed,
                                  v30_m64n64_packed,   v30_m64n128_packed,
                                  v30_m128n128_packed};
std::array<std::once_flag, 128> initialized;

void configure(int64_t device) {
  TORCH_CHECK(device >= 0 && device < 128, "Invalid CUDA device");
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  std::call_once(initialized[device], [] {
    int stats[6];
    for (auto fn :
         {v26_m256n128_packed_configure, v26_m128n256_packed_configure,
          v29_m128n256_packed_configure, v30_m32n128_packed_configure,
          v30_m64n64_packed_configure, v30_m64n128_packed_configure,
          v30_m128n128_packed_configure, m16n64_packed_configure}) {
      int status = fn(stats);
      TORCH_CHECK(status == 0, "Packed kernel initialization failed: ", status);
    }
  });
}

at::Tensor run(const at::Tensor& x, const at::Tensor& payload,
               const at::Tensor& metadata, at::Tensor workspace, int64_t n,
               int64_t k, int64_t variant, int64_t splits) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
                  x.is_contiguous() && x.size(1) == k,
              "Expected contiguous CUDA FP16 [M,K]");
  TORCH_CHECK(n > 0 && n <= 65536 && n % 8 == 0 && k > 0 && k % 64 == 0 &&
                  x.size(0) <= 1048576,
              "Unsupported dimensions");
  TORCH_CHECK(
      payload.device() == x.device() && payload.scalar_type() == at::kByte &&
          payload.is_contiguous() && payload.dim() == 1 &&
          payload.numel() >= n * k * 3 / 2 && payload.numel() <= n * k * 2,
      "Invalid packed payload");
  TORCH_CHECK(metadata.device() == x.device() &&
                  metadata.scalar_type() == at::kUInt64 &&
                  metadata.is_contiguous() && metadata.dim() == 2 &&
                  metadata.size(1) == 2 && metadata.size(0) == (n + 255) / 256,
              "Invalid packed group metadata");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0,
              "Unaligned input");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty({x.size(0), n}, x.options());
  if (!x.size(0)) return out;
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  int status;
  if (variant == -2) {
    TORCH_CHECK(x.size(0) == 1, "Native path is M1 only");
    status =
        glm_mixed_gemv_m1(x.data_ptr(), payload.data_ptr(), metadata.data_ptr(),
                          out.data_ptr(), n, k, 0, stream);
  } else if (variant == -1) {
    TORCH_CHECK(x.size(0) <= 512 && splits >= 1 && splits <= 8 &&
                    workspace.device() == x.device() &&
                    workspace.scalar_type() == at::kFloat &&
                    workspace.is_contiguous() &&
                    workspace.numel() >= splits * x.size(0) * n,
                "Invalid split-K workspace");
    status = m16n64_packed(
        x.data_ptr(), payload.data_ptr(), metadata.data_ptr(), out.data_ptr(),
        workspace.data_ptr(), x.size(0), n, k, splits, stream);
  } else {
    TORCH_CHECK(variant >= 0 && variant < int64_t(prefill.size()),
                "Invalid prefill variant");
    status =
        prefill[variant](x.data_ptr(), payload.data_ptr(), metadata.data_ptr(),
                         out.data_ptr(), x.size(0), n, k, stream);
  }
  TORCH_CHECK(status == 0, "Packed dense CUDA launch failed: ", status);
  return out;
}
}  // namespace

TORCH_LIBRARY(glm_packed_dense, m) {
  m.def("configure(int device) -> ()", configure);
  m.def(
      "run(Tensor x, Tensor payload, Tensor metadata, Tensor(a!) workspace, "
      "int n, int k, int variant, int splits) -> Tensor");
}
TORCH_LIBRARY_IMPL(glm_packed_dense, CUDA, m) { m.impl("run", run); }
