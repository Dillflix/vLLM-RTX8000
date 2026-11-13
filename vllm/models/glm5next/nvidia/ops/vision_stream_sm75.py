# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU-fed GLM vision chunks with unchanged complete-frame attention.

This changes GEMM batch shapes, which can change FP16 rounding. It never removes
patches, splits a spatial attention sequence, changes normalization or projection
math, or resamples media. A temporal patch larger than the chunk budget remains
whole; the budget is not a maximum supported image/video resolution.
"""

import torch


def validate_vision_stream(vllm_config) -> bool:
    enabled = vllm_config.additional_config.get("glm_sm75_stream_vision", False)
    shard = vllm_config.additional_config.get("glm_sm75_shard_encoder_cache", False)
    if type(enabled) is not bool:
        raise TypeError("glm_sm75_stream_vision must be a boolean")
    if type(shard) is not bool:
        raise TypeError("glm_sm75_shard_encoder_cache must be a boolean")
    if shard and not enabled:
        raise ValueError("Sharded encoder storage requires glm_sm75_stream_vision")
    if not enabled:
        return False

    from vllm.platforms import current_platform

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    mm = model.multimodal_config
    compile_config = vllm_config.compilation_config
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
        or mm is None
        or mm.mm_encoder_tp_mode != "weights"
        or mm.is_multimodal_pruning_enabled()
        or compile_config.cudagraph_mm_encoder
        or compile_config.compile_mm_encoder
        or vllm_config.ec_transfer_config is not None
    ):
        raise ValueError(
            "glm_sm75_stream_vision requires SM75 FP16 GLM TP2/PP2, encoder "
            "weight TP, and no DBO, CP/DP, encoder graphs, pruning or EC transfer"
        )
    return True


def frame_chunk_plan(grid_thw, *, max_patches: int = 32768):
    """Return original-feature chunks in order; never split a temporal patch."""
    if type(max_patches) is not int or max_patches <= 0:
        raise ValueError("Vision chunk patch budget must be a positive integer")
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("Expected an N-by-3 temporal/spatial grid")
        grid_thw = grid_thw.tolist()
    result = []
    offset = 0
    for grid in grid_thw:
        if len(grid) != 3 or any(type(value) is not int for value in grid):
            raise ValueError("Vision grid values must be integer triplets")
        t, h, w = grid
        if min(t, h, w) <= 0 or h % 2 or w % 2:
            raise ValueError("GLM vision requires positive, spatially even grids")
        frame_patches = h * w
        frames_per_chunk = max(1, max_patches // frame_patches)
        chunks = []
        for first in range(0, t, frames_per_chunk):
            count = min(frames_per_chunk, t - first)
            start = offset + first * frame_patches
            end = start + count * frame_patches
            chunks.append((start, end, [count, h, w]))
        result.append((t * frame_patches // 4, chunks))
        offset += t * frame_patches
    if not result:
        raise ValueError("At least one image or video feature is required")
    return offset, result


@torch.inference_mode()
def stream_vision_features(tower, pixels, grid_thw, vllm_config, *, max_patches=32768):
    """Produce one ordinary tensor (or opt-in large-feature shard) per item."""
    total_patches, plan = frame_chunk_plan(grid_thw, max_patches=max_patches)
    if (
        pixels.ndim != 2
        or pixels.shape[0] != total_patches
        or pixels.device.type != "cpu"
        or not pixels.is_contiguous()
        or tower.spatial_merge_size != 2
    ):
        raise ValueError("Streaming vision requires contiguous CPU patches and merge2")
    shard = vllm_config.additional_config.get("glm_sm75_shard_encoder_cache", False)
    if type(shard) is not bool:
        raise TypeError("glm_sm75_shard_encoder_cache must be a boolean")
    if total_patches <= max_patches and not (
        shard and any(output_rows > 8000 for output_rows, _ in plan)
    ):
        # Preserve the original GEMM batch shape and launch count for ordinary
        # single/multi-image batches that already fit the transient budget.
        output = tower(pixels, grid_thw)
        return output.split([output_rows for output_rows, _ in plan])
    outputs = []
    for output_rows, chunks in plan:
        builder = None
        if shard and output_rows > 8000:
            from .encoder_cache_tp import make_glm_encoder_cache_builder

            builder = make_glm_encoder_cache_builder(
                vllm_config,
                expected_rows=output_rows,
                hidden_size=tower.out_hidden_size,
            )
        result = None
        if builder is None and len(chunks) > 1:
            # One full-width result allocation, not a full-width list plus cat.
            result = torch.empty(
                (output_rows, tower.out_hidden_size),
                dtype=tower.dtype,
                device=tower.device,
            )
        copied_rows = 0
        for start, end, grid in chunks:
            # The original tower moves/normalizes only this CPU view on device.
            # All layers and the final TP reduction run unchanged.
            chunk = tower(pixels[start:end], [grid])
            expected = (end - start) // 4
            if chunk.shape != (expected, tower.out_hidden_size):
                raise RuntimeError("Vision chunk output does not match original grid")
            if builder is not None:
                builder.append(chunk)
            elif result is not None:
                result[copied_rows : copied_rows + expected].copy_(chunk)
            else:
                result = chunk
            copied_rows += expected
            del chunk
        assert copied_rows == output_rows
        outputs.append(builder.finish() if builder is not None else result)
    return tuple(outputs)
