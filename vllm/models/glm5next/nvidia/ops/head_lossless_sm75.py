# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-weight GLM output-head storage for the qualified SM75 profile.

The original ParallelLMHead stays in place so the MTP drafter shares its module,
vocabulary metadata and packed storage. Qualified native arithmetic12 plans
cover decode/verification rows. Larger calls reconstruct one bounded stripe.
"""

import hashlib
import importlib
import types
import weakref
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from vllm import envs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.workspace import current_workspace_manager

from .arithmetic12_codec import pack_rows, unpack_rows
from .arithmetic12_tileplanes import row12_to_tileplanes, tileplanes_to_row12
from .dense_sm75 import DenseHead, DenseLinear

HEAD_SHAPE = (77440, 4096)
STRIPE_BYTES = 64 * 1024 * 1024
# Both real head shards passed M1..15 numerical/graph/performance gates.
# Variant0 is the adjacent-plane N64/K64 native MMA implementation.
HEAD_PLANS: dict[int, int] = {rows: 0 for rows in range(1, 16)}


def _hidden_weight_aliases(head, pointer):
    """Inspect layer/method fields and closures without retaining any tensor."""
    aliases, visited = [], set()

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            if (
                value.is_cuda
                and value.untyped_storage().data_ptr() == pointer
                and path != "head._parameters.weight"
            ):
                aliases.append(path)
            return
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, types.FunctionType):
            for index, cell in enumerate(value.__closure__ or ()):
                visit(cell.cell_contents, f"{path}.closure[{index}]")
            visit(value.__defaults__, f"{path}.defaults")
        elif isinstance(value, types.MethodType):
            visit(value.__self__, f"{path}.self")
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            visit(vars(value), path)

    visit(head, "head")
    return aliases


def validate_configuration(vllm_config) -> bool:
    enabled = vllm_config.additional_config.get("glm_sm75_lossless_head", False)
    if type(enabled) is not bool:
        raise ValueError("glm_sm75_lossless_head must be a boolean")
    if not enabled:
        return False
    from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled
    from vllm.platforms import current_platform

    model = vllm_config.model_config
    hf = model.hf_text_config
    parallel = vllm_config.parallel_config
    speculative = vllm_config.speculative_config
    offload = vllm_config.offload_config
    if (
        model.dtype != torch.float16
        or model.head_dtype not in (None, torch.float16)
        or current_platform.get_device_capability() != (7, 5)
        or parallel.tensor_parallel_size != 2
        or parallel.pipeline_parallel_size != 2
        or parallel.enable_dbo
        or hf.model_type != "glm5_next_text"
        or hf.hidden_size != HEAD_SHAPE[1]
        or getattr(hf, "tie_word_embeddings", False)
        or model.word_embeddings_untied_by_checkpoint
        or model.enable_sleep_mode
        or vllm_config.lora_config is not None
        or vllm_config.compilation_config.mode != 0
        or vllm_config.kernel_config.linear_backend != "auto"
        or envs.VLLM_BATCH_INVARIANT
        or offload.uva.cpu_offload_gb > 0
        or offload.prefetch.offload_group_size > 0
        or direct_reload_enabled()
        or speculative is None
        or speculative.method != "mtp"
        or (
            speculative is not None
            and (
                speculative.draft_lm_head_quantization is not None
                or speculative.draft_model_config.head_dtype
                not in (None, torch.float16)
            )
        )
    ):
        raise ValueError(
            "Lossless GLM head requires SM75 FP16 TP2/PP2, untied embeddings, "
            "FP16 head dtype, mode0/auto linear, no DBO/LoRA/offload/reload/sleep "
            "or batch-invariant mode, and an unquantized shared MTP head"
        )
    return True


@triton.jit
def _unpack12_stripe(
    P, OUT, COUNT, N: tl.constexpr, K: tl.constexpr, ROW_START, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < COUNT
    row = ROW_START + offset // K
    column = offset % K
    first_n = row // 64 * 64
    tile_rows = tl.minimum(64, N - first_n)
    first_k = column // 64 * 64
    local = (row % 64) * 64 + column % 64
    base = (first_n * K + first_k * tile_rows) * 3 // 2
    low = tl.load(P + base + local, valid, other=0).to(tl.uint32)
    high = tl.load(P + base + tile_rows * 64 + local // 2, valid, other=0)
    high = high.to(tl.uint32)
    code = low | (((high >> ((local % 2) * 4)) & 15) << 8)
    payload = code & 2047
    shift = tl.minimum(3, tl.maximum((payload.to(tl.int32) >> 7) - 1, 0))
    bits = (((payload - (shift << 7)) << shift) | ((code & 2048) << 4)).to(tl.uint16)
    tl.store(OUT + offset, bits.to(tl.float16, bitcast=True), valid)


def unpack_stripe(packed, output, row_start):
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 2
        or output.dtype != torch.float16
        or output.ndim != 2
        or not packed.is_cuda
        or packed.device != output.device
        or not packed.is_contiguous()
        or not output.is_contiguous()
        or output.shape[1] * 3 != packed.shape[1] * 2
        or output.shape[1] % 64 != 0
        or row_start < 0
        or row_start + output.shape[0] > packed.shape[0]
        or output.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    ):
        raise ValueError("Invalid arithmetic12 head stripe or overlapping storage")
    if output.numel():
        _unpack12_stripe[(triton.cdiv(output.numel(), 256),)](
            packed,
            output,
            output.numel(),
            packed.shape[0],
            output.shape[1],
            row_start,
            BLOCK=256,
        )


def _stripe_workspace(flat, packed, bias):
    stripe_rows = STRIPE_BYTES // (HEAD_SHAPE[1] * 2)
    (scratch,) = current_workspace_manager().get_simultaneous(
        ((stripe_rows, HEAD_SHAPE[1]), torch.float16)
    )
    if any(
        value is not None and torch._C._overlaps(scratch, value)
        for value in (flat, packed, bias)
    ):
        # Preserve the previously supported alias case. This exceptional private
        # 64MiB allocation is observable by the untimed worker audit, and freed
        # after the same-stream stripe consumers complete.
        scratch = torch.empty_like(scratch)
    return scratch


def _packed_logits(
    x: torch.Tensor,
    packed: torch.Tensor,
    variant: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        x.dtype != torch.float16
        or not x.is_cuda
        or packed.device != x.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or tuple(packed.shape) != (HEAD_SHAPE[0], HEAD_SHAPE[1] * 3 // 2)
        or x.ndim < 1
        or x.shape[-1] != HEAD_SHAPE[1]
        or (
            bias is not None
            and (
                bias.dtype != x.dtype
                or bias.device != x.device
                or bias.shape != (HEAD_SHAPE[0],)
            )
        )
    ):
        raise ValueError("Invalid FP16 GLM output-head operands")
    flat = x.reshape(-1, HEAD_SHAPE[1])
    output = x.new_empty((flat.shape[0], HEAD_SHAPE[0]))
    if flat.shape[0] == 0:
        return output.reshape(*x.shape[:-1], HEAD_SHAPE[0])
    if variant >= 0:
        if not (
            1 <= flat.shape[0] <= 15
            and flat.is_contiguous()
            and flat.data_ptr() % 16 == 0
            and bias is None
        ):
            raise ValueError("Native arithmetic12 plans require contiguous M1..15")
        torch.ops._rtx8000_C.lossless_head(flat, packed, output, variant)
    else:
        stripe_rows = STRIPE_BYTES // (HEAD_SHAPE[1] * 2)
        scratch = _stripe_workspace(flat, packed, bias)
        # Same-stream consumers finish before the next stripe overwrites scratch.
        # Output is independent storage and may outlive all later workspace users.
        for start in range(0, HEAD_SHAPE[0], stripe_rows):
            end = min(start + stripe_rows, HEAD_SHAPE[0])
            weight = scratch[: end - start]
            unpack_stripe(packed, weight, start)
            stripe_bias = None if bias is None else bias[start:end]
            output[:, start:end].copy_(F.linear(flat, weight, stripe_bias))
    return output.reshape(*x.shape[:-1], HEAD_SHAPE[0])


def _packed_logits_fake(
    x: torch.Tensor,
    packed: torch.Tensor,
    variant: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], packed.shape[0]))


direct_register_custom_op(
    op_name="glm_sm75_lossless_head12",
    op_func=_packed_logits,
    fake_impl=_packed_logits_fake,
)


class PackedHeadMethod(QuantizeMethodBase):
    """Not an Unquantized* method: incompatible head_dtype cannot cast .weight."""

    def __init__(self, plans):
        self.plans = dict(plans)

    def create_weights(self, layer, *args, **kwargs):
        raise RuntimeError("PackedHeadMethod is installed only after weight loading")

    def process_weights_after_loading(self, layer):
        raise RuntimeError("Reloading an installed packed output head is unsupported")

    def tie_weights(self, layer, embed_tokens):
        raise RuntimeError("A packed output head cannot be retied to embeddings")

    def apply(self, layer, x, bias=None):
        rows = x.numel() // HEAD_SHAPE[1] if x.ndim else 0
        variant = (
            self.plans.get(rows, -1)
            if x.is_contiguous() and x.data_ptr() % 16 == 0 and bias is None
            else -1
        )
        return torch.ops.vllm.glm_sm75_lossless_head12(
            x, layer.lossless_head_codes, variant, bias
        )


def install(model) -> list[dict]:
    """Pack only the target-owned head; never replace or copy its module."""
    if not validate_configuration(model.vllm_config):
        return []
    head = model.lm_head
    if not isinstance(head, ParallelLMHead):
        # PP0 owns no head and must allocate no head storage or fallback scratch.
        from vllm.model_executor.models.utils import PPMissingLayer

        if not isinstance(head, PPMissingLayer):
            raise TypeError("Unexpected GLM pipeline output-head ownership")
        return []
    if set(HEAD_PLANS) != set(range(1, 16)):
        raise RuntimeError("Native head M1..15 performance plans are not qualified")
    if model.logits_processor.head_dtype not in (None, torch.float16):
        raise ValueError("Packed GLM output head requires FP16 head_dtype")
    method = head.quant_method
    if type(method) in (DenseHead, DenseLinear):
        method = cast(DenseHead | DenseLinear, method).original
    if (
        type(method) not in (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        or head.tp_size != 2
        or head.has_bias
        or hasattr(head, "lossless_head_codes")
        or head.weight.dtype != torch.float16
        or tuple(head.weight.shape) != HEAD_SHAPE
        or not head.weight.is_cuda
        or not head.weight.is_contiguous()
    ):
        raise ValueError("Only the unquantized FP16 GLM TP2 output head is eligible")
    pointer = head.weight.untyped_storage().data_ptr()
    hidden_aliases = _hidden_weight_aliases(head, pointer)
    if hidden_aliases:
        raise ValueError(f"Output-head weight has cached aliases: {hidden_aliases}")
    for name, value in [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]:
        if (
            name != "lm_head.weight"
            and value.device == head.weight.device
            and value.untyped_storage().data_ptr() == pointer
        ):
            raise ValueError(f"Output-head weight aliases {name}")
    # No GPU reference to the old weight survives this local inventory.
    del value
    importlib.import_module("vllm._rtx8000_C")
    if not hasattr(torch.ops._rtx8000_C, "lossless_head"):
        raise RuntimeError("The ordinary RTX8000 extension lacks lossless_head")
    original = head.weight.detach().cpu().contiguous()
    words = original.view(torch.uint16).numpy()
    packed = torch.empty((HEAD_SHAPE[0], HEAD_SHAPE[1] * 3 // 2), dtype=torch.uint8)
    for start in range(0, HEAD_SHAPE[0], 256):
        block = words[start : start + 256]
        codes = pack_rows(block, 12)
        if not np.array_equal(unpack_rows(codes, 12), block):
            raise RuntimeError("Arithmetic12 output-head reconstruction is not exact")
        tiles = row12_to_tileplanes(codes)
        restored_codes = tileplanes_to_row12(tiles, codes.shape)
        if not np.array_equal(restored_codes, codes):
            raise RuntimeError("Tile-plane output-head reconstruction is not exact")
        packed[start : start + len(block)].copy_(
            torch.from_numpy(tiles.reshape(codes.shape))
        )
    record = {
        "name": "lm_head",
        "shape": list(HEAD_SHAPE),
        "storage_format": "arithmetic12_n64_k64_adjacent_planes_v1",
        "original_bytes": original.numel() * original.element_size(),
        "packed_bytes": packed.numel(),
        "verified_elements": original.numel(),
        "original_words_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
        "packed_sha256": hashlib.sha256(packed.numpy().tobytes()).hexdigest(),
        "plans": dict(HEAD_PLANS),
        "fallback_max_scratch_bytes": STRIPE_BYTES,
        "tp_rank": head.tp_rank,
        "vocab_shard": vars(head.shard_indices),
        "preinstall_hidden_weight_aliases": hidden_aliases,
        "original_gpu_pointer": pointer,
    }
    device = head.weight.device
    head._lossless_head_retired_weight_ref = weakref.ref(head.weight)
    # Free the original GPU parameter before copying the verified packed buffer.
    # A failed device copy aborts this initialization; no live service is mutated.
    del head.weight
    head.register_buffer("lossless_head_codes", packed.to(device))
    head.quant_method = PackedHeadMethod(HEAD_PLANS)
    record["original_parameter_retired_before_return"] = (
        head._lossless_head_retired_weight_ref() is None
    )
    # Reserve via the same shared arena used by attention; never persistent.
    current_workspace_manager().get_simultaneous(
        ((STRIPE_BYTES // (HEAD_SHAPE[1] * 2), HEAD_SHAPE[1]), torch.float16)
    )
    return [record]
