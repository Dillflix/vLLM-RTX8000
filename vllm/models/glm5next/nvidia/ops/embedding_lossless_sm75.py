# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental exact12-bit FP16 embedding lookup for GLM on SM75.

Only dictionary indices are packed. The lookup loads original16-bit words and
bitcasts them to FP16; it performs no floating-point arithmetic. Vocabulary
ownership and the existing TP reduction remain identical to the ordinary layer.
The output head is not eligible for this storage method.
"""

import numpy as np
import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .embedding12_codec import make_dictionary, pack_rows, unpack_rows


def validate_scope(
    *,
    enabled,
    dtype,
    capability,
    tensor_parallel_size,
    pipeline_parallel_size,
    model_type,
    hidden_size,
    tie_word_embeddings,
    has_lora,
    has_cpu_offload,
    has_direct_reload,
) -> bool:
    """Fail closed outside the tested storage/ownership contract."""
    if type(enabled) is not bool:
        raise ValueError("glm_sm75_lossless_embedding must be a boolean")
    if not enabled:
        return False
    if (
        dtype != torch.float16
        or capability != (7, 5)
        or tensor_parallel_size != 2
        or pipeline_parallel_size != 2
        or model_type != "glm5_next_text"
        or hidden_size != 4096
        or tie_word_embeddings
        or has_lora
        or has_cpu_offload
        or has_direct_reload
    ):
        raise ValueError(
            "Lossless GLM embedding requires SM75 FP16 TP2/PP2, hidden size4096, "
            "untied input embeddings, no LoRA/offload and no weight reloading"
        )
    return True


def validate_configuration(vllm_config) -> bool:
    """Validate actual target/draft config without constructing a GPU layer."""
    enabled = vllm_config.additional_config.get("glm_sm75_lossless_embedding", False)
    if type(enabled) is not bool:
        raise ValueError("glm_sm75_lossless_embedding must be a boolean")
    if not enabled:
        return False
    from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled
    from vllm.platforms import current_platform

    model = vllm_config.model_config
    hf = model.hf_text_config
    return validate_scope(
        enabled=enabled,
        dtype=model.dtype,
        capability=current_platform.get_device_capability(),
        tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
        pipeline_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
        model_type=hf.model_type,
        hidden_size=hf.hidden_size,
        tie_word_embeddings=(
            getattr(hf, "tie_word_embeddings", False)
            or model.word_embeddings_untied_by_checkpoint
        ),
        has_lora=vllm_config.lora_config is not None,
        has_cpu_offload=(
            vllm_config.offload_config.uva.cpu_offload_gb > 0
            or vllm_config.offload_config.prefetch.offload_group_size > 0
        ),
        has_direct_reload=direct_reload_enabled(),
    )


def make_embedding(vllm_config, *args, **kwargs):
    """GLM constructors can select this factory without changing normal loading."""
    constructor = (
        LosslessVocabParallelEmbedding
        if validate_configuration(vllm_config)
        else VocabParallelEmbedding
    )
    return constructor(*args, **kwargs)


def install(model: torch.nn.Module) -> list[dict]:
    """Finalize only input embeddings, after ordinary loading and weight tying.

    Called by the owning GLM model's post-load hook. Loader finalization runs
    this after quant methods and maybe_retie_word_embeddings. Any registered
    parameter/buffer storage alias (including a shared output head) is rejected
    before the first mutation.
    """
    candidates = [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, LosslessVocabParallelEmbedding)
    ]
    registered = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    for name, layer in candidates:
        if hasattr(layer, "lossless_codes"):
            raise RuntimeError("Reloading a compressed embedding is not supported")
        if not name.endswith("embed_tokens") or isinstance(layer, ParallelLMHead):
            raise ValueError("Only explicitly named GLM input embeddings are eligible")
        storage = layer.weight.untyped_storage()
        aliases = [
            key
            for key, parameter in registered
            if parameter.device == layer.weight.device
            and parameter.untyped_storage().data_ptr() == storage.data_ptr()
            and key != f"{name}.weight"
        ]
        if aliases:
            raise ValueError(f"Input embedding storage is shared with {aliases}")
    # Do not retain a Python reference to the old GPU parameter during/after
    # compression; the above inventory exists solely for alias validation.
    del registered
    if candidates:
        del storage
    return [
        dict(name=name, **layer.compress_loaded_weight()) for name, layer in candidates
    ]


@triton.jit
def _lookup12(
    ids,
    packed,
    dictionary,
    output,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    DICTIONARY_SIZE: tl.constexpr,
    ORG_START: tl.constexpr,
    ORG_END: tl.constexpr,
    ORG_PADDING: tl.constexpr,
    ADDED_START: tl.constexpr,
    ADDED_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(ids + row).to(tl.int64)
    original = (token >= ORG_START) & (token < ORG_END)
    added = (token >= ADDED_START) & (token < ADDED_END)
    added_offset = ADDED_START - (ORG_END - ORG_START) - ORG_PADDING
    local_row = tl.where(
        original, token - ORG_START, tl.where(added, token - added_offset, 0)
    )
    valid = (original | added) & (col < WIDTH)
    triplet = local_row * (WIDTH // 2 * 3) + (col // 2) * 3
    low_or_high = tl.load(
        packed + triplet + tl.where((col & 1) == 0, 0, 2), mask=valid, other=0
    ).to(tl.uint32)
    middle = tl.load(packed + triplet + 1, mask=valid, other=0).to(tl.uint32)
    code = tl.where(
        (col & 1) == 0,
        low_or_high | ((middle & 15) << 8),
        (middle >> 4) | (low_or_high << 4),
    )
    bits = tl.load(dictionary + code, mask=valid & (code < DICTIONARY_SIZE), other=0)
    # Reinterpret16 bits. No fp16 conversion, addition, FMA or rescaling.
    value = bits.to(tl.float16, bitcast=True)
    tl.store(output + row * WIDTH + col, value, mask=col < WIDTH)


def lookup12(
    input_ids: torch.Tensor,
    packed: torch.Tensor,
    dictionary: torch.Tensor,
    width: int,
    org_start: int,
    org_end: int,
    org_padding: int,
    added_start: int,
    added_end: int,
    block: int = 256,
) -> torch.Tensor:
    if (
        input_ids.dtype not in (torch.int32, torch.int64)
        or packed.dtype != torch.uint8
        or dictionary.dtype != torch.uint16
        or not input_ids.is_cuda
        or packed.device != input_ids.device
        or dictionary.device != input_ids.device
        or not input_ids.is_contiguous()
        or not packed.is_contiguous()
        or not dictionary.is_contiguous()
        or packed.ndim != 2
        or dictionary.ndim != 1
        or not 0 < dictionary.numel() <= 4096
        or width <= 0
        or width % 2
        or packed.shape[1] != width // 2 * 3
        or block not in (128, 256, 512, 1024)
        or min(org_start, org_end, org_padding, added_start, added_end) < 0
        or org_end < org_start
        or added_end < added_start
        or org_end - org_start + org_padding + added_end - added_start > packed.shape[0]
    ):
        raise ValueError(
            "Invalid exact12-bit embedding layout, IDs or vocabulary shard"
        )
    out = torch.empty(
        (*input_ids.shape, width), dtype=torch.float16, device=input_ids.device
    )
    rows = input_ids.numel()
    if rows:
        _lookup12[(rows, triton.cdiv(width, block))](
            input_ids,
            packed,
            dictionary,
            out,
            rows,
            width,
            dictionary.numel(),
            org_start,
            org_end,
            org_padding,
            added_start,
            added_end,
            BLOCK=block,
            num_warps=4,
        )
    return out


def _lookup12_fake(
    input_ids: torch.Tensor,
    packed: torch.Tensor,
    dictionary: torch.Tensor,
    width: int,
    org_start: int,
    org_end: int,
    org_padding: int,
    added_start: int,
    added_end: int,
    block: int = 256,
) -> torch.Tensor:
    return torch.empty(
        (*input_ids.shape, width), dtype=torch.float16, device=input_ids.device
    )


direct_register_custom_op(
    op_name="glm_sm75_lossless_embedding12",
    op_func=lookup12,
    fake_impl=_lookup12_fake,
)


class LosslessVocabParallelEmbedding(VocabParallelEmbedding):
    """Ordinary loading/ownership followed by an explicit post-load conversion."""

    def compress_loaded_weight(self):
        if (
            isinstance(self, ParallelLMHead)
            or type(self.quant_method) is not UnquantizedEmbeddingMethod
            or self.tp_size != 2
            or self.params_dtype != torch.float16
            or self.embedding_dim != 4096
            or hasattr(self, "lossless_codes")
            or self.weight.dtype != torch.float16
            or not self.weight.is_cuda
            or torch.cuda.get_device_capability(self.weight.device) != (7, 5)
        ):
            raise ValueError(
                "Only an ordinary FP16 TP2 GLM input embedding is eligible"
            )
        cpu = self.weight.detach().cpu().contiguous()
        words = cpu.view(torch.uint16).numpy()
        dictionary = make_dictionary(words)
        # Build/verify bounded chunks so packing does not require another full
        # uint16 inverse tensor in addition to the loaded FP16 shard.
        packed = torch.empty(
            (cpu.shape[0], self.embedding_dim // 2 * 3), dtype=torch.uint8
        )
        for start in range(0, cpu.shape[0], 256):
            bits = words[start : start + 256]
            codes = pack_rows(bits, dictionary)
            if not np.array_equal(unpack_rows(codes, dictionary), bits):
                raise RuntimeError(
                    "Exact embedding reconstruction failed before installation"
                )
            packed[start : start + bits.shape[0]].copy_(torch.from_numpy(codes))
        old_bytes = self.weight.untyped_storage().nbytes()
        device = self.weight.device
        self.register_buffer("lossless_codes", packed.to(device))
        self.register_buffer(
            "lossless_dictionary", torch.from_numpy(dictionary).to(device)
        )
        self.lossless_original_bytes = old_bytes
        self.lossless_verified_elements = cpu.numel()
        del self.weight
        return {
            "original_bytes": old_bytes,
            "packed_bytes": self.lossless_codes.untyped_storage().nbytes(),
            "dictionary_bytes": self.lossless_dictionary.untyped_storage().nbytes(),
            "verified_elements": self.lossless_verified_elements,
        }

    def forward(self, input_):
        if not hasattr(self, "lossless_codes"):
            return super().forward(input_)
        indices = self.shard_indices
        result = torch.ops.vllm.glm_sm75_lossless_embedding12(
            input_,
            self.lossless_codes,
            self.lossless_dictionary,
            self.embedding_dim,
            indices.org_vocab_start_index,
            indices.org_vocab_end_index,
            indices.num_org_vocab_padding,
            indices.added_vocab_start_index,
            indices.added_vocab_end_index,
        )
        if result.numel() == 0:
            return result
        if self.reduce_results:
            result = (
                self.parallel_group.all_reduce(result)
                if self.parallel_group is not None
                else tensor_model_parallel_all_reduce(result)
            )
        return result
