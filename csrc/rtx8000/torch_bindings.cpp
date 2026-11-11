// SPDX-License-Identifier: Apache-2.0
#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cstdint>
#include "launch.h"

namespace {
void check_device(const at::Tensor& x, const at::Tensor& other) {
  TORCH_CHECK(other.is_cuda() && other.device() == x.device(),
              "SM75 tensors must share a CUDA device");
}
void sparse75(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
              const at::Tensor& indices, const at::Tensor& table,
              const at::Tensor& requests, const at::Tensor& gate,
              at::Tensor out, at::Tensor partial, at::Tensor lse,
              int64_t splits, int64_t variant) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 3 &&
                  q.size(1) == 12 && q.size(2) == 256 && q.stride(1) == 256 &&
                  q.stride(2) == 1 && q.stride(0) >= 3072,
              "SM75 QSA requires FP16 Q [rows,12,256]");
  c10::cuda::CUDAGuard guard(q.device());
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5, "SM75 QSA requires Turing");
  for (const auto& t :
       {k, v, indices, table, requests, gate, out, partial, lse})
    check_device(q, t);
  TORCH_CHECK(
      k.dim() == 4 && k.size(0) > 0 && k.size(1) > 0 && k.size(2) == 1 &&
          k.size(3) == 256 && k.scalar_type() == at::kHalf &&
          v.scalar_type() == at::kHalf && v.sizes() == k.sizes() &&
          v.strides() == k.strides() && k.stride(0) >= k.size(1) * 512 &&
          k.stride(1) == 512 && k.stride(3) == 1 && k.stride(0) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
          v.data_ptr<at::Half>() == k.data_ptr<at::Half>() + 256,
      "SM75 QSA requires packed FP16 K/V");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0) && indices.size(1) > 1 &&
                  indices.stride(1) == 1 && table.scalar_type() == at::kInt &&
                  table.dim() == 2 && table.size(0) > 0 && table.size(1) > 0 &&
                  table.stride(1) == 1 && requests.scalar_type() == at::kInt &&
                  requests.dim() == 1 && requests.size(0) == q.size(0) &&
                  requests.is_contiguous(),
              "Invalid QSA selection metadata");
  for (const auto& t : {gate, out})
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.sizes() == q.sizes() &&
                    t.stride(0) >= 3072 && t.stride(1) == 256 &&
                    t.stride(2) == 1,
                "Invalid QSA output/gate");
  TORCH_CHECK(splits == 1 || splits == 4 || splits == 16 || splits == 32,
              "Unsupported QSA split count");
  bool valid_variant = variant == 0;
#define RTX_QSA_VARIANT(ID, NAME, KEYS, WARPS, PREFETCH) \
  valid_variant = valid_variant || variant == ID;
#include "qsa_variants.inc"
#undef RTX_QSA_VARIANT
  TORCH_CHECK(valid_variant, "Unsupported QSA tile variant");
  if (splits > 1) {
    TORCH_CHECK(partial.scalar_type() == at::kFloat &&
                    partial.is_contiguous() &&
                    partial.numel() >= splits * q.numel() &&
                    lse.scalar_type() == at::kFloat && lse.is_contiguous() &&
                    lse.numel() >= splits * q.size(0) * 12,
                "Insufficient QSA FP32 workspace");
  }
  if (!q.size(0)) return;
  Qsa75Shape shape{
      int(q.size(0)),     int(k.size(0)),           int(k.size(1)),
      int(table.size(1)), int(table.stride(0)),     int(table.size(0)),
      int(splits),        int(indices.size(1) - 1), int(indices.stride(0)),
      int(q.stride(0)),   int(out.stride(0)),       int(gate.stride(0)),
      k.stride(0)};
  C10_CUDA_CHECK(static_cast<cudaError_t>(sparse75_variant_launch(
      q.data_ptr(), k.data_ptr(), indices.data_ptr(), table.data_ptr(),
      requests.data_ptr(), gate.data_ptr(), out.data_ptr(), partial.data_ptr(),
      lse.data_ptr(), shape, at::cuda::getCurrentCUDAStream(), int(variant))));
}

void dense75(const at::Tensor& x, const at::Tensor& w, at::Tensor out,
             int64_t variant) {
  TORCH_CHECK(
      x.is_cuda() && x.dim() == 2 && w.dim() == 2 && out.dim() == 2 &&
          x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf &&
          out.scalar_type() == at::kHalf && x.is_contiguous() &&
          w.is_contiguous() && out.is_contiguous() && x.size(1) == w.size(1) &&
          out.size(0) == x.size(0) && out.size(1) == w.size(0) &&
          x.size(1) > 0 && w.size(0) > 0 && x.size(1) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 &&
          variant >= 0 && variant <= 11,
      "Invalid SM75 dense inputs");
  check_device(x, w);
  check_device(x, out);
  c10::cuda::CUDAGuard guard(x.device());
  auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 7 && prop->minor == 5,
              "SM75 dense requires Turing");
  TORCH_CHECK(dense75_launch(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                             x.size(0), w.size(0), x.size(1), variant,
                             at::cuda::getCurrentCUDAStream()) == 0,
              "SM75 dense launch failed");
}
}  // namespace
TORCH_LIBRARY(_rtx8000_C, m) {
  m.def(
      "qsa(Tensor q, Tensor k, Tensor v, Tensor indices, Tensor table, Tensor "
      "requests, Tensor gate, Tensor(a!) out, Tensor(b!) partial, Tensor(c!) "
      "lse, int splits, int variant=0) -> ()");
  m.def("dense(Tensor x, Tensor weight, Tensor(a!) out, int variant) -> ()");
}
TORCH_LIBRARY_IMPL(_rtx8000_C, CUDA, m) {
  m.impl("qsa", sparse75);
  m.impl("dense", dense75);
}
PyMODINIT_FUNC PyInit__rtx8000_C() {
  static PyModuleDef def = {PyModuleDef_HEAD_INIT, "_rtx8000_C", nullptr, -1,
                            nullptr};
  return PyModule_Create(&def);
}
