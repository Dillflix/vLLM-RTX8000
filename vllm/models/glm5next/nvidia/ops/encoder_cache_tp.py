# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact TP-column storage for already-reduced GLM encoder output chunks.

This module does not change encoder arithmetic. A producer must append complete
post-TP-reduction chunks in feature order, then release its full-width chunks.
Only the column shard is retained. Consumers reconstruct the requested rows by
all-gather, without reductions, casts, pruning, or transfers to host memory.

The storage primitive accepts CPU tensors for real distributed CPU tests. Its
production factory is restricted to SM75 FP16 GLM, TP2/PP2 and the first PP
stage. Encoder graph capture, pruning, DBO, and EC transfer are unsupported.
"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol

import torch


class ColumnGather(Protocol):
    world_size: int
    rank_in_group: int

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...


@dataclass(frozen=True)
class TPShardedEncoderOutput:
    chunks: tuple[torch.Tensor, ...]
    offsets: tuple[int, ...]
    hidden_size: int
    max_slice_rows: int
    group: ColumnGather

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.offsets[-1], self.hidden_size))

    @property
    def dtype(self) -> torch.dtype:
        return self.chunks[0].dtype

    @property
    def device(self) -> torch.device:
        return self.chunks[0].device

    @property
    def storage_bytes(self) -> int:
        return sum(chunk.untyped_storage().nbytes() for chunk in self.chunks)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Encoder cache supports contiguous row slices only")
        start = 0 if key.start is None else key.start
        end = len(self) if key.stop is None else key.stop
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("Encoder cache slice bounds must be Python integers")
        if not 0 <= start <= end <= len(self):
            raise IndexError("Encoder cache row range is outside the feature")
        if end - start > self.max_slice_rows:
            raise ValueError("Encoder cache row range exceeds the LM step budget")
        if start == end:
            return self.chunks[0].new_empty((0, self.hidden_size))
        first = bisect_right(self.offsets, start) - 1
        views = []
        for index in range(first, len(self.chunks)):
            lo, hi = self.offsets[index : index + 2]
            if lo >= end:
                break
            views.append(
                self.chunks[index][max(start - lo, 0) : min(end - lo, hi - lo)]
            )
        local = views[0] if len(views) == 1 else torch.cat(views, dim=0)
        return self.group.all_gather(local, dim=-1)


class TPShardedEncoderBuilder:
    """Store half of each bounded, fully computed encoder output chunk."""

    def __init__(
        self,
        *,
        expected_rows: int,
        hidden_size: int,
        max_slice_rows: int,
        group: ColumnGather,
    ):
        if group.world_size != 2 or group.rank_in_group not in (0, 1):
            raise ValueError("GLM encoder cache requires a two-rank TP group")
        if expected_rows <= 0 or hidden_size <= 0 or hidden_size % 2:
            raise ValueError("Encoder output must have positive rows and even width")
        if max_slice_rows <= 0:
            raise ValueError("The LM slice budget must be positive")
        self.expected_rows = expected_rows
        self.hidden_size = hidden_size
        self.max_slice_rows = max_slice_rows
        self.group = group
        self._chunks: list[torch.Tensor] = []
        self._offsets = [0]
        self._finished = False

    @torch.inference_mode()
    def append(self, output: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("Encoder output builder is already finished")
        if (
            output.ndim != 2
            or output.shape[1] != self.hidden_size
            or output.shape[0] <= 0
            or output.dtype != torch.float16
        ):
            raise ValueError("Expected a non-empty full-width FP16 output chunk")
        if self._chunks and output.device != self._chunks[0].device:
            raise ValueError("All output chunks must be on the same device")
        end = self._offsets[-1] + output.shape[0]
        if end > self.expected_rows:
            raise ValueError("Output chunks exceed this feature's token count")
        width = self.hidden_size // 2
        start = self.group.rank_in_group * width
        # Clone is essential: a narrow view would retain the full allocation.
        shard = output[:, start : start + width].clone(
            memory_format=torch.contiguous_format
        )
        self._chunks.append(shard)
        self._offsets.append(end)

    def finish(self) -> TPShardedEncoderOutput:
        if self._finished:
            raise RuntimeError("Encoder output builder is already finished")
        if self._offsets[-1] != self.expected_rows:
            raise ValueError("Incomplete encoder feature; missing output chunks")
        self._finished = True
        result = TPShardedEncoderOutput(
            chunks=tuple(self._chunks),
            offsets=tuple(self._offsets),
            hidden_size=self.hidden_size,
            max_slice_rows=self.max_slice_rows,
            group=self.group,
        )
        self._chunks.clear()
        return result


def read_encoder_cache_slice(
    output: torch.Tensor | TPShardedEncoderOutput,
    start: int,
    end: int,
    reconstructed: dict[tuple[int, int, int], torch.Tensor],
) -> torch.Tensor:
    """Reuse identical reconstruction only within one embedding-gather call.

    The caller must create and discard `reconstructed` inside each invocation
    of EncoderRunner.gather_mm_embeddings. Values must never be retained across
    scheduling steps, cache eviction, or a changed feature. Normal tensor cache
    entries keep their existing zero-copy view behavior.
    """
    if not isinstance(output, TPShardedEncoderOutput):
        return output[start:end]
    key = (id(output), start, end)
    if key not in reconstructed:
        reconstructed[key] = output[start:end]
    return reconstructed[key]


def make_glm_encoder_cache_builder(
    vllm_config, *, expected_rows: int, hidden_size: int
) -> TPShardedEncoderBuilder | None:
    """Production entry point; disabled by default and fail closed on mismatch."""
    enabled = vllm_config.additional_config.get("glm_sm75_shard_encoder_cache", False)
    if not isinstance(enabled, bool):
        raise TypeError("glm_sm75_shard_encoder_cache must be a boolean")
    if not enabled:
        return None

    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.platforms import current_platform

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    multimodal = model.multimodal_config
    compilation = vllm_config.compilation_config
    if (
        getattr(model.hf_config, "model_type", None) != "glm5_next"
        or model.dtype != torch.float16
        or not current_platform.is_device_capability(75)
        or parallel.tensor_parallel_size != 2
        or parallel.pipeline_parallel_size != 2
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.use_ubatching
        or not get_pp_group().is_first_rank
        or multimodal is None
        or multimodal.mm_encoder_tp_mode != "weights"
        or multimodal.is_multimodal_pruning_enabled()
        or compilation.cudagraph_mm_encoder
        or compilation.compile_mm_encoder
        or vllm_config.ec_transfer_config is not None
    ):
        raise ValueError(
            "glm_sm75_shard_encoder_cache requires SM75 FP16 GLM TP2/PP2 on "
            "PP0 with encoder weight TP; no DBO, CP/DP, encoder graphs, pruning "
            "or EC transfer"
        )
    return TPShardedEncoderBuilder(
        expected_rows=expected_rows,
        hidden_size=hidden_size,
        max_slice_rows=(
            vllm_config.scheduler_config.max_num_batched_tokens
            + vllm_config.num_speculative_tokens
            * vllm_config.scheduler_config.max_num_seqs
        ),
        group=get_tp_group(),
    )
