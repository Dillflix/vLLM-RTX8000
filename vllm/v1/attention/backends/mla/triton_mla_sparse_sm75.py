# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FP16 sparse MLA qualification backend for SM75."""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import round_up
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.triton_mla_sparse_sm75 import (
    _BLOCK_H,
    _MIN_PREFILL_H32_TOKENS,
    _choose_num_kv_splits,
    triton_mla_sparse_attention,
)


@dataclass
class TritonMLASparseSM75Metadata(XPUMLASparseMetadata):
    # CPU query spans; inherited decode counts intentionally route all rows to
    # sparse MQA and cannot distinguish the true prefill/decode work here.
    large_prefill_spans: tuple[tuple[int, int], ...] = ()


class TritonMLASparseSM75MetadataBuilder(XPUMLASparseMetadataBuilder):
    # Graph support is enabled only after full-model capture qualification.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.model_config.hf_text_config
        # GLM appends the incomplete pool's raw-token tail after its top-k pools.
        # Match the model's allocation; truncating to index_topk drops this tail.
        self.topk_tokens = round_up(config.index_topk + config.index_kpool - 1, 128)
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], dtype=torch.int32, device=self.device
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        starts = common_attn_metadata.query_start_loc_cpu.tolist()
        spans = tuple(
            (start, end)
            for start, end in zip(starts, starts[1:])
            if end - start >= _MIN_PREFILL_H32_TOKENS
            and end <= metadata.num_actual_tokens
        )
        return TritonMLASparseSM75Metadata(**vars(metadata), large_prefill_spans=spans)


class TritonMLASparseSM75Impl(XPUMLASparseImpl):
    supports_pcp = False
    # The inherited builder routes every token through sparse MQA.
    supports_dense_mha_prefill = False
    profiles_prefill_workspace = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_size != 512 or self.kv_lora_rank != 512:
            raise ValueError("SM75 sparse MLA currently requires NoPE-512")
        cfg = get_current_vllm_config_or_none()
        self.prefill_h32 = (
            cast(dict[str, Any], cfg.additional_config).get(
                "glm_sm75_prefill_h32", False
            )
            if cfg is not None
            else False
        )
        if not isinstance(self.prefill_h32, bool):
            raise ValueError("glm_sm75_prefill_h32 must be a boolean")
        if self.prefill_h32 and (
            cfg is None
            or not current_platform.is_device_capability(75)
            or cfg.parallel_config.decode_context_parallel_size != 1
            or cfg.parallel_config.prefill_context_parallel_size != 1
            or (
                cfg.model_config is not None
                and (
                    cfg.model_config.dtype != torch.float16
                    or cfg.model_config.hf_text_config.model_type != "glm5_next_text"
                )
            )
        ):
            raise ValueError(
                "glm_sm75_prefill_h32 requires SM75 FP16 GLM with PCP=DCP=1"
            )
        self.nope_query_view = (
            cast(dict[str, Any], cfg.additional_config).get(
                "glm_sm75_mla_nope_view", False
            )
            if cfg is not None
            else False
        )
        if not isinstance(self.nope_query_view, bool):
            raise ValueError("glm_sm75_mla_nope_view must be a boolean")
        if self.nope_query_view and (
            cfg is None
            or not current_platform.is_device_capability(75)
            or cfg.model_config is None
            or cfg.model_config.dtype != torch.float16
            or cfg.parallel_config.decode_context_parallel_size != 1
            or cfg.parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError("glm_sm75_mla_nope_view requires SM75 FP16 PCP=DCP=1")
        self._sm_count = (
            num_compute_units(self.topk_indices_buffer.device.index)
            if self.topk_indices_buffer is not None
            else None
        )

    def profile_workspace_bytes(self, max_tokens: int) -> int:
        """Bound extra live buffers, excluding the caller's Q/output and KV.

        Common MLA allocates absorbed latent Q; inherited forward_mqa currently
        concatenates even an empty NoPE tail, and attention allocates latent V.
        All three can coexist with the remapped int32 indices and split scratch.
        The indexer's persistent top-k/workspace arena is already profiled.
        """
        if self.topk_indices_buffer is None or self._sm_count is None:
            raise RuntimeError("SM75 MLA profiling requires its bound top-k buffer")
        topk = self.topk_indices_buffer.shape[1]
        groups = (self.num_heads + _BLOCK_H - 1) // _BLOCK_H
        peak = 0
        # A small decode can allocate split scratch although a large prefill
        # does not. Take the peak of actual shapes, not their sum.
        for rows in range(1, max_tokens + 1):
            splits = _choose_num_kv_splits(rows, groups, topk, self._sm_count)
            latent = 3 * rows * self.num_heads * self.kv_lora_rank * 2
            indices = rows * topk * 4
            partials = (
                rows * self.num_heads * splits * (self.kv_lora_rank + 1) * 4
                if splits > 1
                else 0
            )
            peak = max(peak, latent + indices + partials)
        return peak

    def reserve_workspace(self) -> None:
        cfg = get_current_vllm_config_or_none()
        if cfg is None or self.topk_indices_buffer is None:
            raise RuntimeError("SM75 MLA profiling requires the active model config")
        max_tokens = cfg.scheduler_config.max_num_batched_tokens
        if max_tokens > self.topk_indices_buffer.shape[0]:
            raise ValueError("SM75 MLA top-k buffer is smaller than the token budget")
        # Transient allocation, like the generic dense-MHA profile simulation.
        # It contributes to peak accounting without retaining extra GPU storage.
        _ = torch.empty(
            self.profile_workspace_bytes(max_tokens),
            dtype=torch.uint8,
            device=self.topk_indices_buffer.device,
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        query = q[0] if isinstance(q, tuple) else q
        if query.shape[0] == 0:
            return query.new_empty((0, self.num_heads, self.kv_lora_rank)), None
        if (
            self.nope_query_view
            and isinstance(q, tuple)
            and len(q) == 2
            and q[1].shape == (*query.shape[:-1], 0)
            and q[1].dtype == query.dtype
            and q[1].device == query.device
            and query.stride(-1) == 1
        ):
            # NoPE has no rotary columns to append. Preserve the exact latent
            # words and supported token/head strides; avoid the inherited cat.
            # Unsupported last-dimension strides retain the original copy path.
            q = query
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        # The inherited method name denotes unquantized KV, including FP16.
        kv = kv_c_and_k_pe_cache.view(-1, 1, 512)
        indices = topk_indices.view(q.shape[0], 1, topk_indices.shape[-1])
        spans = (
            getattr(attn_metadata, "large_prefill_spans", ())
            if self.prefill_h32 and self.num_heads == 32
            else ()
        )
        if spans:
            output = q.new_empty((q.shape[0], self.num_heads, self.kv_lora_rank))
            cursor = 0
            for start, end in spans:
                if (
                    not (cursor <= start < end <= q.shape[0])
                    or end - start < _MIN_PREFILL_H32_TOKENS
                ):
                    raise ValueError("Invalid SM75 large-prefill query spans")
                if start > cursor:
                    triton_mla_sparse_attention(
                        q[cursor:start],
                        kv,
                        indices[cursor:start],
                        sm_scale=self.softmax_scale,
                        sm_count=self._sm_count,
                        num_kv_splits=1,
                        out=output[cursor:start],
                    )
                triton_mla_sparse_attention(
                    q[start:end],
                    kv,
                    indices[start:end],
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    prefill_h32=True,
                    out=output[start:end],
                )
                cursor = end
            if cursor < q.shape[0]:
                triton_mla_sparse_attention(
                    q[cursor:],
                    kv,
                    indices[cursor:],
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    num_kv_splits=1,
                    out=output[cursor:],
                )
            return output
        return triton_mla_sparse_attention(
            q,
            kv,
            indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )


class TritonMLASparseSM75Backend(XPUMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_SM75"

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # The inherited builder constructs request IDs from CPU boundaries.
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7 and capability.minor == 5

    @staticmethod
    def get_builder_cls() -> type[TritonMLASparseSM75MetadataBuilder]:
        return TritonMLASparseSM75MetadataBuilder

    @staticmethod
    def get_metadata_cls() -> type[TritonMLASparseSM75Metadata]:
        return TritonMLASparseSM75Metadata

    @staticmethod
    def get_impl_cls() -> type[TritonMLASparseSM75Impl]:
        return TritonMLASparseSM75Impl
