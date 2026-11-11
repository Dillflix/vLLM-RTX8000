// SPDX-License-Identifier: Apache-2.0
// The companion uses two guarded kernels for its qualified geometry. The
// original MoE operator and its full kernel library remain unchanged.
#define MARLIN_NAMESPACE_NAME rtx8000_marlin
#define MARLIN_EXTERNAL_SELECTOR 1
#define MARLIN_FORCE_EXACT_SHARED_MEM 1
#define MARLIN_GEMM_ENTRY_NAME rtx8000_moe_wna16_marlin_gemm
#define MARLIN_COMPANION_REGISTRATION 1
#define MARLIN_RTX8000_GUARDED_SELECTOR 1

#include "ops.cu"
#include "core/registration.h"

STABLE_TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  m.def(
      "rtx8000_moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("rtx8000_moe_wna16_marlin_gemm",
         TORCH_BOX(&rtx8000_moe_wna16_marlin_gemm));
}

PyMODINIT_FUNC PyInit__rtx8000_marlin_C() {
  static PyModuleDef definition = {PyModuleDef_HEAD_INIT, "_rtx8000_marlin_C",
                                   nullptr, 0, nullptr};
  PyObject* module = PyModule_Create(&definition);
  if (module == nullptr) return nullptr;
  if (PyModule_AddIntConstant(module, "MARLIN_POLICY_REVISION", 2) < 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}
