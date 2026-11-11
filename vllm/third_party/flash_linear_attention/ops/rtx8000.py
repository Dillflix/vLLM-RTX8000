# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM75 kernels for the measured Flash-Next GDN geometry."""

import importlib
from functools import lru_cache

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def _is_sm75(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 5)


def _base_shape(k, v, g, cu_seqlens, chunk_size):
    if (
        not k.is_cuda
        or not _is_sm75(k.device.index)
        or k.ndim != 4
        or v.ndim != 4
        or k.shape[0] != 1
        or k.shape[2:] != (8, 128)
        or v.shape != (1, k.shape[1], 24, 128)
        or chunk_size != 64
        or g is None
        or g.shape != (1, k.shape[1], 24)
        or g.dtype != torch.float32
    ):
        return False
    if (
        any(t.device != k.device or not t.is_contiguous() for t in (k, v, g))
        or k.dtype != v.dtype
        or k.dtype != torch.float16
    ):
        return False
    return cu_seqlens is None or (
        cu_seqlens.device == k.device
        and cu_seqlens.dtype in (torch.int32, torch.int64)
        and cu_seqlens.ndim == 1
        and cu_seqlens.numel() >= 2
        and cu_seqlens.is_contiguous()
    )


def use_state_tiles(
    k,
    w,
    u,
    g,
    gk,
    initial_state,
    cu_seqlens,
    chunk_size,
    output_final_state,
    save_new_value,
    use_exp2,
):
    if not envs.VLLM_RTX8000_GDN_TILES or not _base_shape(
        k, u, g, cu_seqlens, chunk_size
    ):
        return False
    return (
        512 <= k.shape[1] <= 2048
        and (cu_seqlens is None or cu_seqlens.numel() == 2)
        and w.shape == (1, k.shape[1], 24, 128)
        and w.dtype == torch.float16
        and w.device == k.device
        and w.is_contiguous()
        and initial_state is not None
        and initial_state.shape == (1, 24, 128, 128)
        and initial_state.dtype == torch.float32
        and initial_state.device == k.device
        and initial_state.is_contiguous()
        and gk is None
        and output_final_state
        and save_new_value
        and not use_exp2
    )


def use_output_tiles(q, k, v, h, g, cu_seqlens, chunk_size):
    if not envs.VLLM_RTX8000_GDN_TILES or not _base_shape(
        k, v, g, cu_seqlens, chunk_size
    ):
        return False
    return (
        512 <= k.shape[1] <= 2048
        and (cu_seqlens is None or cu_seqlens.numel() == 2)
        and q.shape == k.shape
        and q.dtype == h.dtype == torch.float16
        and q.device == h.device == k.device
        and q.is_contiguous()
        and h.is_contiguous()
        and h.ndim == 5
        and h.shape[0] == 1
        and h.shape[2:] == (24, 128, 128)
    )


@lru_cache(maxsize=None)
def report_tiles(kind):
    logger.info("RTX8000 GDN %s tile dispatch active", kind)


@lru_cache(maxsize=1)
def _load_native():
    importlib.import_module("vllm._rtx8000_C")
    logger.info("Native SM75 GDN kernels loaded")


def try_native_state(
    k,
    w,
    u,
    g,
    gk,
    initial_state,
    cu_seqlens,
    chunk_offsets,
    h,
    v_new,
    final_state,
    chunk_size,
    use_exp2,
):
    if not envs.VLLM_RTX8000_NATIVE_GDN_H or not _base_shape(
        k, u, g, cu_seqlens, chunk_size
    ):
        return False
    if (
        gk is not None
        or use_exp2
        or cu_seqlens is None
        or chunk_offsets is None
        or initial_state is None
        or v_new is None
        or final_state is None
        or w.shape != u.shape
        or w.dtype != torch.float16
        or initial_state.shape != (cu_seqlens.numel() - 1, 24, 128, 128)
        or initial_state.dtype != torch.float32
        or chunk_offsets.shape != cu_seqlens.shape
        or chunk_offsets.dtype not in (torch.int32, torch.int64)
        or any(
            t.device != k.device or not t.is_contiguous()
            for t in (w, initial_state, chunk_offsets)
        )
        or any(t.data_ptr() % 16 for t in (k, w))
    ):
        return False
    _load_native()
    report_tiles("native state V16")
    torch.ops._rtx8000_C.gdn_h(
        k,
        w,
        u,
        g,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        h,
        v_new,
        final_state,
        1,
    )
    return True


def try_native_output(q, k, v, h, g, cu_seqlens, chunk_indices, out, scale, chunk_size):
    if not envs.VLLM_RTX8000_NATIVE_GDN or not _base_shape(
        k, v, g, cu_seqlens, chunk_size
    ):
        return False
    if (
        k.shape[1] == 0
        or cu_seqlens is None
        or chunk_indices is None
        or q.shape != k.shape
        or q.dtype != torch.float16
        or h.dtype != torch.float16
        or out.dtype != torch.float16
        or h.ndim != 5
        or h.shape[0] != 1
        or h.shape[2:] != (24, 128, 128)
        or chunk_indices.shape != (h.shape[1], 2)
        or chunk_indices.dtype not in (torch.int32, torch.int64)
        or chunk_indices.dtype != cu_seqlens.dtype
        or out.shape != v.shape
        or any(
            t.device != k.device or not t.is_contiguous()
            for t in (q, h, out, chunk_indices)
        )
        or any(t.data_ptr() % 32 for t in (q, k, v, h))
    ):
        return False
    _load_native()
    torch.ops._rtx8000_C.gdn_o(
        q, k, v, h, g, cu_seqlens, chunk_indices, out, float(scale), 0
    )
    return True
