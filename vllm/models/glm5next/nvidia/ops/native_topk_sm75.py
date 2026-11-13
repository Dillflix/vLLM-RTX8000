# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 GLM prefill score/top-k with bounded shared scratch.

This adapter deliberately supports only the qualified H32/D128 FP8-query,
lossless FP16-key scratch, FP32-scale contracts.
The caller retains short-context/small-row and
all paged decode kernels. The RTX8000 extension build installs the native library.
"""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

MIN_ROWS = 128
MIN_WIDTH = 16384


def use_native(rows: int, width: int) -> bool:
    return rows >= MIN_ROWS and width >= MIN_WIDTH


def stage_sets(width: int) -> list[int]:
    sets = (width + 4095) // 4096
    result = [sets]
    while sets > 8:
        sets = (sets + 7) // 8
        result.append(sets)
    return result


def scratch_elements(max_rows: int, max_width: int) -> int:
    """FP32 elements; native rank-key stages alias the fallback logits buffer."""
    native = max_rows * sum(stage_sets(max_width)) * 512 * 2
    short = max_rows * min(max_width, MIN_WIDTH - 1)
    few_rows = min(max_rows, MIN_ROWS - 1) * max_width
    # The rank-key view needs an even number of FP32 elements.
    return (max(native, short, few_rows, max_width) + 1) // 2 * 2


@lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libglm_indexer.so")))
    for name in ("glm_local_runtime", "glm_local_fp8decoded_runtime"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        fn.restype = ctypes.c_int
    lib.glm_merge.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    )
    lib.glm_merge.restype = ctypes.c_int
    return lib


def select(
    q, keys, scales, weights, starts, ends, width, scratch, out, *, fp8_arithmetic=False
):
    rows = q.shape[0]
    inputs = (q, keys, scales, weights, starts, ends, scratch, out)
    if not all(x.device == q.device and x.is_contiguous() for x in inputs):
        raise ValueError("Native GLM indexer requires contiguous same-device tensors")
    if (
        q.shape != (rows, 32, 128)
        or q.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or keys.ndim != 2
        or keys.shape[1] != 128
        or keys.dtype != torch.float16
        or not isinstance(fp8_arithmetic, bool)
        or scales.shape != (keys.shape[0],)
        or scales.dtype != torch.float32
        or weights.shape != (rows, 32)
        or weights.dtype != torch.float32
        or starts.shape != (rows,)
        or ends.shape != (rows,)
        or starts.dtype != torch.int32
        or ends.dtype != torch.int32
        or out.shape != (rows, 512)
        or out.dtype != torch.int32
        or scratch.dtype != torch.float32
        or not q.is_cuda
        or rows <= 0
        or width <= 0
    ):
        raise ValueError("Unsupported native GLM indexer contract")
    sets = stage_sets(width)
    count = rows * sum(sets) * 512
    if count * 2 > scratch.numel():
        raise RuntimeError("Native GLM top-k exceeds profiled workspace")
    rank_keys = scratch[: count * 2].view(torch.int64)
    stages = []
    offset = 0
    for n in sets:
        size = rows * n * 512
        stages.append(rank_keys[offset : offset + size].view(rows, n, 512))
        offset += size
    lib = library()
    with torch.accelerator.device_index(q.device.index):
        stream = torch.cuda.current_stream(q.device).cuda_stream
        local = (
            lib.glm_local_fp8decoded_runtime
            if fp8_arithmetic
            else lib.glm_local_runtime
        )
        err = local(
            *[
                x.data_ptr()
                for x in (q, keys, scales, weights, starts, ends, stages[0])
            ],
            rows,
            keys.shape[0],
            width,
            stream,
        )
        if err:
            raise RuntimeError(f"Native GLM local selection CUDA error {err}")
        for i, src in enumerate(stages):
            final = i + 1 == len(stages)
            err = lib.glm_merge(
                src.data_ptr(),
                None if final else stages[i + 1].data_ptr(),
                out.data_ptr() if final else None,
                rows,
                src.shape[1],
                stream,
            )
            if err:
                raise RuntimeError(f"Native GLM top-k merge CUDA error {err}")
    return out
