# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental, opt-in GLM FP16 dense kernels for SM75 TP2/PP2.

The shortlist passed real-input component checks, not whole-model qualification.
Only the named unquantized attention projections and the output head are eligible.
Weights, collectives, and unselected calls retain their existing implementation.
"""

import importlib

import torch

from vllm import envs
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.utils.torch_utils import direct_register_custom_op

# Captured inputs (M2/M4 sliced from M5); both timing orders improved >=3%.
# M3 uses the same vector kernel in the ordinary RTX8000 native extension.
DENSE_PLANS = {
    (12576, 4096): {1: 0},
    (4096, 4096): {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    (4096, 8192): {1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
    (77440, 4096): {1: 0},
}


# Actual MTP verification and draft-prefill captures, not sliced decode rows.
# M10, input projections and shared head retain their existing methods.
MTP_TARGET_PLANS = {
    (4096, 4096): {6: 1, 8: 0},
    (4096, 8192): {6: 1, 8: 0},
}
MTP_DRAFT_PLANS = {(4096, 8192): {2: 0, 4: 1, 8: 0}}


def validate_mtp_configuration(vllm_config) -> bool:
    enabled = vllm_config.additional_config.get("glm_native_dense_mtp", False)
    if type(enabled) is not bool:
        raise ValueError("glm_native_dense_mtp must be a boolean")
    if not enabled:
        return False
    speculative = vllm_config.speculative_config
    if (
        vllm_config.additional_config.get("glm_native_dense") is not True
        or speculative is None
        or speculative.method != "mtp"
        or speculative.num_speculative_tokens != 1
    ):
        raise ValueError("glm_native_dense_mtp requires glm_native_dense and MTP1")
    from vllm.platforms import current_platform

    validate_configuration(
        dtype=vllm_config.model_config.dtype,
        capability=current_platform.get_device_capability(),
        tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
        pipeline_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
        model_type=vllm_config.model_config.hf_text_config.model_type,
        compilation_mode=vllm_config.compilation_config.mode,
        linear_backend=vllm_config.kernel_config.linear_backend,
        has_lora=vllm_config.lora_config is not None,
    )
    return True


def validate_configuration(
    *,
    dtype,
    capability,
    tensor_parallel_size,
    pipeline_parallel_size,
    model_type,
    compilation_mode,
    linear_backend,
    has_lora,
) -> None:
    if (
        dtype != torch.float16
        or capability != (7, 5)
        or tensor_parallel_size != 2
        or pipeline_parallel_size != 2
        or model_type != "glm5_next_text"
    ):
        raise ValueError("glm_native_dense requires GLM FP16 on SM75 with TP2/PP2")
    if compilation_mode != 0 or linear_backend != "auto" or has_lora:
        raise ValueError(
            "glm_native_dense currently requires compilation mode 0, "
            "linear_backend='auto', and no LoRA"
        )


def _dense(x: torch.Tensor, weight: torch.Tensor, variant: int) -> torch.Tensor:
    out = x.new_empty((x.shape[0], weight.shape[0]))
    torch.ops._rtx8000_C.dense(x, weight, out, variant)
    return out


def _dense_fake(x: torch.Tensor, weight: torch.Tensor, variant: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(
    op_name="glm_sm75_native_dense",
    op_func=_dense,
    fake_impl=_dense_fake,
)


class _DenseApply:
    plans: dict[int, int]
    original: UnquantizedLinearMethod | UnquantizedEmbeddingMethod

    def apply(self, layer, x, bias=None):
        variant = self.plans.get(x.shape[0]) if x.ndim == 2 else None
        weight = layer.weight
        if (
            variant is not None
            and bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and x.dtype == weight.dtype == torch.float16
            and x.is_cuda
            and x.device == weight.device
            and x.is_contiguous()
            and weight.is_contiguous()
            and x.data_ptr() % 16 == 0
            and weight.data_ptr() % 16 == 0
        ):
            return torch.ops.vllm.glm_sm75_native_dense(x, weight, variant)
        return self.original.apply(layer, x, bias)

    def process_weights_after_loading(self, layer):
        self.original.process_weights_after_loading(layer)


class DenseLinear(_DenseApply, UnquantizedLinearMethod):
    def __init__(self, original, plans):
        super().__init__()
        self.original, self.plans = original, plans


class DenseHead(_DenseApply, UnquantizedEmbeddingMethod):
    def __init__(self, original, plans):
        self.original, self.plans = original, plans


def install(model, *, mtp: bool = False) -> list[dict]:
    """Called once by the GLM model after weights load, before graph capture."""
    importlib.import_module("vllm._rtx8000_C")
    installed = []
    for name, layer in model.named_modules():
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        shape = tuple(weight.shape)
        if shape not in DENSE_PLANS or weight.dtype != torch.float16:
            continue
        method = getattr(layer, "quant_method", None)
        is_head = (
            name == "lm_head"
            and shape == (77440, 4096)
            and isinstance(layer, ParallelLMHead)
            and type(method) in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        )
        is_projection = (
            name.startswith("model.layers.")
            and name.endswith((".self_attn.in_proj_qkvbfg_a", ".self_attn.o_proj"))
            and shape != (77440, 4096)
            and isinstance(layer, LinearBase)
            and type(method) is UnquantizedLinearMethod
        )
        if not (is_head or is_projection):
            continue
        wrapper = (
            DenseHead if type(method) is UnquantizedEmbeddingMethod else DenseLinear
        )
        plans = dict(DENSE_PLANS[shape])
        if mtp and is_projection and name.endswith(".self_attn.o_proj"):
            plans.update(MTP_TARGET_PLANS.get(shape, {}))
        layer.quant_method = wrapper(method, plans)
        installed.append({"name": name, "shape": list(shape), "rows": plans})
    return installed


def install_mtp_draft(model) -> list[dict]:
    """Wrap only the qualified, unquantized draft output projection.

    The target-owned shared head and the draft checkpoint's quantized experts
    are deliberately outside this shortlist. No parameters or config are copied.
    """
    importlib.import_module("vllm._rtx8000_C")
    installed = []
    for name, layer in model.named_modules():
        if not (
            name.startswith("model.layers.")
            and name.endswith(".mtp_block.self_attn.o_proj")
            and isinstance(layer, LinearBase)
            and type(getattr(layer, "quant_method", None)) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float16:
            continue
        shape = tuple(weight.shape)
        if shape not in MTP_DRAFT_PLANS:
            continue
        plans = dict(MTP_DRAFT_PLANS[shape])
        layer.quant_method = DenseLinear(layer.quant_method, plans)
        installed.append({"name": name, "shape": list(shape), "rows": plans})
    return installed
