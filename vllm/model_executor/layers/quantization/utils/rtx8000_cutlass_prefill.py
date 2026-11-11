# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM75 dense prefill experiment for Qwen Flash-Next INT8 experts."""

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def maybe_cutlass_prefill_method(
    layer: LinearBase, prefix: str
) -> "RTX8000CutlassPrefillMethod | None":
    if not envs.VLLM_RTX8000_CUTLASS_PREFILL:
        return None
    if not prefix.endswith((".linear_attn.in_proj_qkvz", ".linear_attn.out_proj")):
        return None
    if layer.params_dtype != torch.float16:
        raise ValueError("RTX8000 CUTLASS prefill requires FP16 activations")
    if current_platform.get_device_capability() != (7, 5):
        raise ValueError("RTX8000 CUTLASS prefill is qualified only for SM75")
    return RTX8000CutlassPrefillMethod(prefix)


@register_weight_loader_v2_supported_method
class RTX8000CutlassPrefillMethod(UnquantizedLinearMethod):
    """Retain FP16 decode weights and cache one INT8 copy for prefill."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.attention_prefix = prefix.rsplit(".", 1)[0]
        self.prefill_calls = 0
        self.prefill_tokens = 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if any(d % 16 for d in layer.weight.shape):
            raise ValueError("CUTLASS prefill weight dimensions must divide by 16")
        quantized, scales, _ = ops.scaled_int8_quant(layer.weight.contiguous())
        layer.register_buffer("cutlass_prefill_weight", quantized.T)
        layer.register_buffer("cutlass_prefill_scale", scales)
        logger.info(
            "RTX8000 CUTLASS prefill cached %s shape=%s extra_bytes=%d",
            self.prefix,
            tuple(layer.weight.shape),
            quantized.numel() + scales.numel() * scales.element_size(),
        )

    def apply(self, layer, x, bias=None):
        metadata = None
        if is_forward_context_available():
            entries = get_forward_context().attn_metadata
            if isinstance(entries, dict):
                metadata = cast(
                    "GDNAttentionMetadata | None", entries.get(self.attention_prefix)
                )
        if (
            metadata is None
            or metadata.num_prefill_tokens < 128
            or metadata.num_spec_decodes
            or metadata.spec_sequence_masks is not None
        ):
            return super().apply(layer, x, bias)

        flat = x.reshape(-1, x.shape[-1])
        start = metadata.num_decode_tokens
        end = start + metadata.num_prefill_tokens
        if end > flat.shape[0]:
            raise ValueError("GDN prefill metadata exceeds projection input")
        a, scales, _ = ops.scaled_int8_quant(flat[start:end].contiguous())
        prefill = ops.cutlass_scaled_mm(
            a,
            layer.cutlass_prefill_weight,
            scales,
            layer.cutlass_prefill_scale,
            torch.float16,
            bias,
        )
        if self.prefill_calls == 0:
            logger.info("RTX8000 CUTLASS prefill executing %s", self.prefix)
        self.prefill_calls += 1
        self.prefill_tokens += end - start
        if start == 0 and end == flat.shape[0]:
            result = prefill
        else:
            parts = []
            if start:
                parts.append(super().apply(layer, flat[:start], bias))
            parts.append(prefill)
            if end < flat.shape[0]:
                parts.append(super().apply(layer, flat[end:], bias))
            result = torch.cat(parts, dim=0)
        return result.reshape(*x.shape[:-1], result.shape[-1])
