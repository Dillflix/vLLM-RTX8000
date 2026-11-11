# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM75 FP16 kernels qualified on Flash-Next TP2 PP2 serving."""

import importlib
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def load_native() -> None:
    importlib.import_module("vllm._rtx8000_C")
    logger.info("Loaded native SM75 RTX8000 attention/dense kernels")


# Local TP2 shapes; keep cuBLAS for points without measured wins.
# 0/1 = FP32-accumulating vectorized CUDA, 4/8 warps respectively.
# 8 = FP32-accumulating WMMA, BM16/BN64/BK128, shared-memory skew8.
# M8/12/16 additions use captured checkpoint inputs and both RTX8000 boards.
DENSE_PLANS = {
    (48, 2560): {1: 0, 2: 0, 4: 0},
    (512, 2560): {1: 0, 2: 0, 4: 1, 5: 1},
    (640, 2560): {1: 1, 2: 1, 4: 1, 5: 1, 8: 0},
    (2560, 320): {1: 0, 2: 0},
    (7296, 2560): {1: 0, 2: 0, 4: 0, 5: 0, 8: 0, 12: 0, 16: 8},
    (8192, 2560): {1: 0, 2: 0, 4: 0, 5: 0, 8: 0, 12: 8, 16: 8},
    (10240, 320): {1: 0, 2: 0},
    (124160, 2560): {12: 8, 16: 8},
}


def dense(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    variant = DENSE_PLANS.get(tuple(weight.shape), {}).get(x.shape[0] if x.ndim else 0)
    if (
        variant is None
        or x.ndim != 2
        or x.dtype != torch.float16
        or weight.dtype != torch.float16
        or not x.is_cuda
        or not weight.is_cuda
        or x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.data_ptr() % 16
        or weight.data_ptr() % 16
    ):
        return None
    load_native()
    out = x.new_empty((x.shape[0], weight.shape[0]))
    torch.ops._rtx8000_C.dense(x, weight, out, variant)
    return out


def supports_qsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
) -> bool:
    return (
        q.ndim == out.ndim == 3
        and k.ndim == v.ndim == 4
        and q.dtype == k.dtype == v.dtype == out.dtype == torch.float16
        and q.is_cuda
        and q.device == k.device == v.device == out.device
        and out.shape == q.shape
        and q.shape[1:] == (12, 256)
        and v.shape == k.shape
        and k.shape[2:] == (1, 256)
        and q.stride(0) >= 3072
        and q.stride(1) == out.stride(1) == 256
        and q.stride(2) == out.stride(2) == 1
        and k.stride(0) >= k.shape[1] * 512
        and k.stride(0) % 8 == 0
        and k.stride(1) == 512
        and k.stride(3) == 1
        and k.data_ptr() % 16 == 0
        and out.stride(0) >= 3072
        and v.stride() == k.stride()
        and v.data_ptr() == k.data_ptr() + 512
    )


# (prefetched-V variant, key splits), qualified with cold-cache replay on two
# RTX8000s. Keep the original plan for token counts outside the measured set.
QSA_PLANS = {
    1: (4, 32),
    2: (4, 32),
    4: (4, 32),
    5: (4, 32),
    8: (4, 16),
    12: (4, 16),
    16: (4, 16),
    20: (4, 16),
    448: (4, 4),
    512: (4, 4),
    1024: (4, 4),
    1600: (4, 1),
    2048: (4, 1),
}


def sparse(q, k, v, indices, table, requests, out, gate):
    # Split across selected keys at small M to fill Turing's 72 SMs.
    load_native()
    report_layout(tuple(k.shape[1:]), k.stride(), indices.shape[1])
    rows = q.shape[0]
    splits = 32 if rows <= 2 else 16 if rows <= 8 else 4 if rows < 128 else 1
    variant, splits = QSA_PLANS.get(rows, (0, splits))
    if splits == 1:
        partial = lse = out
    else:
        partial = torch.empty((splits, *q.shape), device=q.device, dtype=torch.float32)
        lse = torch.empty((splits, rows, 12), device=q.device, dtype=torch.float32)
    torch.ops._rtx8000_C.qsa(
        q, k, v, indices, table, requests, gate, out, partial, lse, splits, variant
    )
    if splits > 1:
        from .qsa import _qsa_merge_splitk_kernel

        _qsa_merge_splitk_kernel[(rows, 12)](
            partial,
            lse,
            out,
            gate,
            out.stride(0),
            out.stride(1),
            gate.stride(0),
            gate.stride(1),
            rows,
            HEAD_DIM=256,
            NUM_QUERY_HEADS=12,
            NUM_SPLITS=splits,
            BLOCK_SPLITS=splits,
            num_warps=2,
            num_stages=1,
        )
    return out


@lru_cache(maxsize=16)
def report_layout(shape, strides, selection_columns):
    logger.info(
        "Native SM75 QSA active: cache shape=%s strides=%s selection_columns=%d",
        shape,
        strides,
        selection_columns,
    )


@lru_cache(maxsize=16)
def report_fallback(
    query_shape, query_strides, cache_shape, cache_strides, value_offset
):
    logger.warning(
        "Native SM75 QSA fallback: Q=%s/%s K=%s/%s V offset=%d bytes",
        query_shape,
        query_strides,
        cache_shape,
        cache_strides,
        value_offset,
    )
