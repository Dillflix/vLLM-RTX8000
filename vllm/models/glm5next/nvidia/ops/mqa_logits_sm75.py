# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 indexer logits: E4M3 storage, FP16 operands, FP32 accumulation.

Adapted from wtdcode/vllm-backport's Triton MQA fallback, with compressed
prefill rows and per-query decode lengths matching the current vLLM API.
Tensor Core lowering requires the Turing-enabled Triton compiler; stock Triton
can silently lower these FP16 dots to much slower FP32 SIMT instructions.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e4m3fn_to_fp16(value):
    bits = value.to(tl.uint32)
    magnitude = bits & 0x7F
    sign = (bits & 0x80) << 8
    normal = (sign | ((magnitude + 0x40) << 7)).to(tl.uint16)
    normal = normal.to(tl.float16, bitcast=True)
    subnormal = (bits & 7).to(tl.float32) * (1.0 / 512.0)
    subnormal = tl.where(sign != 0, -subnormal, subnormal).to(tl.float16)
    decoded = tl.where(magnitude < 8, subnormal, normal)
    return tl.where(magnitude == 0x7F, float("nan"), decoded).to(tl.float16)


@triton.jit
def _head_sum(q_values, k_bytes, key_scales, weights, Q_FP16: tl.constexpr):
    q = q_values if Q_FP16 else _e4m3fn_to_fp16(q_values)
    k = k_bytes if k_bytes.dtype == tl.float16 else _e4m3fn_to_fp16(k_bytes)
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    scores = scores * key_scales[None, :]
    return tl.sum(tl.maximum(scores, 0.0) * weights[:, None], axis=0)


@triton.jit
def _prefill_logits(
    Q,
    K,
    S,
    W,
    KS,
    KE,
    OUT,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_s,
    stride_w_m,
    stride_w_h,
    stride_ks,
    stride_ke,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    Q_FP16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    start = tl.load(KS + row * stride_ks)
    end = tl.load(KE + row * stride_ke)
    keys = start + cols
    valid = (start >= 0) & (keys < end) & (keys < N) & (cols < WIDTH)
    if tl.sum(valid.to(tl.int32), axis=0) == 0:
        tl.store(OUT + row * WIDTH + cols, float("-inf"), cols < WIDTH)
        return
    heads = tl.arange(0, BH)
    dims = tl.arange(0, BD)
    q = tl.load(
        Q + row * stride_q_m + heads[:, None] * stride_q_h + dims[None, :] * stride_q_d,
        (heads[:, None] < H) & (dims[None, :] < D),
        0,
    )
    k = tl.load(
        K + keys[:, None] * stride_k_n + dims[None, :] * stride_k_d,
        valid[:, None] & (dims[None, :] < D),
        0,
    )
    scales = tl.load(S + keys * stride_s, valid, 0.0)
    weights = tl.load(W + row * stride_w_m + heads * stride_w_h, heads < H, 0.0)
    scores = _head_sum(q, k, scales, weights, Q_FP16)
    scores = tl.where(valid, scores, float("-inf"))
    tl.store(OUT + row * WIDTH + cols, scores, cols < WIDTH)


@triton.jit
def _prefill_logits_dynamic(
    Q,
    K,
    S,
    W,
    KS,
    KE,
    OUT,
    stride_q_m,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_s,
    stride_w_m,
    stride_w_h,
    stride_ks,
    stride_ke,
    N,
    WIDTH,
    H: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    Q_FP16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    start = tl.load(KS + row * stride_ks)
    end = tl.load(KE + row * stride_ke)
    keys = start + cols
    valid = (start >= 0) & (keys < end) & (keys < N) & (cols < WIDTH)
    if tl.sum(valid.to(tl.int32), axis=0) == 0:
        tl.store(OUT + row * WIDTH + cols, float("-inf"), cols < WIDTH)
        return
    heads = tl.arange(0, BH)
    dims = tl.arange(0, BD)
    q = tl.load(
        Q + row * stride_q_m + heads[:, None] * stride_q_h + dims[None, :] * stride_q_d,
        (heads[:, None] < H) & (dims[None, :] < D),
        0,
    )
    k = tl.load(
        K + keys[:, None] * stride_k_n + dims[None, :] * stride_k_d,
        valid[:, None] & (dims[None, :] < D),
        0,
    )
    scales = tl.load(S + keys * stride_s, valid, 0.0)
    weights = tl.load(W + row * stride_w_m + heads * stride_w_h, heads < H, 0.0)
    scores = _head_sum(q, k, scales, weights, Q_FP16)
    scores = tl.where(valid, scores, float("-inf"))
    tl.store(OUT + row * WIDTH + cols, scores, cols < WIDTH)


@triton.jit
def _paged_logits(
    Q,
    KV,
    S,
    W,
    L,
    BT,
    OUT,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_page,
    stride_scale_page,
    stride_w_t,
    stride_w_h,
    stride_l_b,
    stride_l_t,
    stride_bt_b,
    stride_bt_k,
    NEXT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    Q_FP16: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    batch = row // NEXT
    token = row % NEXT
    cols = tile * BN + tl.arange(0, BN)
    logical_pages = cols // PAGE
    offsets = cols % PAGE
    length = tl.load(L + batch * stride_l_b + token * stride_l_t)
    page = tl.load(
        BT + batch * stride_bt_b + logical_pages * stride_bt_k,
        (logical_pages < TABLE_COLS) & (cols < length) & (cols < WIDTH),
        -1,
    ).to(tl.int64)
    valid_page = (page >= 0) & (page < PAGES)
    valid = valid_page & (cols < length) & (cols < WIDTH)
    if tl.sum(valid.to(tl.int32), axis=0) == 0:
        tl.store(
            OUT + row * WIDTH + cols,
            float("-inf"),
            cols < WIDTH,
        )
        return
    heads = tl.arange(0, BH)
    dims = tl.arange(0, BD)
    q = tl.load(
        Q
        + batch * stride_q_b
        + token * stride_q_t
        + heads[:, None] * stride_q_h
        + dims[None, :] * stride_q_d,
        (heads[:, None] < H) & (dims[None, :] < D),
        0,
    )
    k = tl.load(
        KV + page[:, None] * stride_page + offsets[:, None] * D + dims[None, :],
        valid[:, None] & (dims[None, :] < D),
        0,
    )
    scales = tl.load(S + page * stride_scale_page + offsets, valid, 0.0)
    weights = tl.load(W + row * stride_w_t + heads * stride_w_h, heads < H, 0.0)
    scores = _head_sum(q, k, scales, weights, Q_FP16)
    scores = tl.where(valid, scores, float("-inf"))
    tl.store(OUT + row * WIDTH + cols, scores, cols < WIDTH)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("SM75 MQA requires E4M3FN bytes")
    return tensor.view(torch.uint8)


def fp8_mqa_logits_sm75(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    max_seqlen_k: int,
    *,
    dynamic_lengths: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return compressed [query, relative-key] logits with causal padding -inf.

    FP16 queries bypass quantization; their head weights must not include an
    FP8 query scale. Quantized queries require that scale folded into weights.

    `out`, when supplied, must be contiguous FP32 [query, relative-key] storage
    on the query device, disjoint from all inputs, and live through consumption.
    """
    keys, scales = kv
    rows, heads, dim = q.shape
    if dim != 128 or keys.ndim != 2 or keys.shape[1] != dim:
        raise ValueError("SM75 GLM indexer requires 128-dimensional keys")
    if weights.shape != (rows, heads) or weights.dtype != torch.float32:
        raise ValueError("SM75 MQA requires FP32 per-head weights")
    if starts.shape != (rows,) or ends.shape != (rows,):
        raise ValueError("SM75 MQA requires one key interval per query")
    scales = scales.reshape(-1)
    if scales.shape != (keys.shape[0],) or scales.dtype != torch.float32:
        raise ValueError("SM75 MQA requires one FP32 scale per key")
    q = q if q.dtype == torch.float16 else _bytes(q)
    # FP16 is gathered-prefill scratch; paged decode retains FP8 storage.
    keys = keys if keys.dtype == torch.float16 else _bytes(keys)
    if out is None:
        out = torch.empty((rows, max_seqlen_k), dtype=torch.float32, device=q.device)
    elif (
        out.shape != (rows, max_seqlen_k)
        or out.dtype != torch.float32
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "SM75 MQA out must be contiguous FP32 [rows, width] on Q device"
        )
    if rows and max_seqlen_k:
        # Tiny widths can change FP32 contraction when lengths are dynamic.
        # Preserve their original kernel, along with unsupported layouts.
        use_dynamic = (
            dynamic_lengths
            and heads == 32
            and q.is_contiguous()
            and keys.is_contiguous()
            and keys.shape[0] >= 64
            and max_seqlen_k >= 64
        )
        kernel = _prefill_logits_dynamic if use_dynamic else _prefill_logits
        kernel[(rows, triton.cdiv(max_seqlen_k, 64))](
            q,
            keys,
            scales,
            weights,
            starts,
            ends,
            out,
            *q.stride(),
            *keys.stride(),
            scales.stride(0),
            *weights.stride(),
            starts.stride(0),
            ends.stride(0),
            keys.shape[0],
            max_seqlen_k,
            heads,
            dim,
            max(16, triton.next_power_of_2(heads)),
            128,
            64,
            q.dtype == torch.float16,
            num_warps=4,
            num_stages=1,
        )
    return out


def fp8_paged_mqa_logits_sm75(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Read packed pages using an independent causal length for each query."""
    batch, next_n, heads, dim = q.shape
    pages, page_size, one, width = kv_cache.shape
    if dim != 128 or one != 1 or width != dim + 4:
        raise ValueError("SM75 GLM indexer requires 128 FP8 bytes and an FP32 scale")
    if context_lens.shape != (batch, next_n):
        raise ValueError("SM75 paged MQA requires per-query lengths [batch, next_n]")
    if weights.shape != (batch * next_n, heads) or weights.dtype != torch.float32:
        raise ValueError("SM75 MQA requires FP32 per-head weights")
    if kv_cache.dtype != torch.uint8 or kv_cache.stride(-1) != 1:
        raise ValueError("SM75 MQA cache must expose contiguous bytes within pages")
    # Cache is page-planar: all K bytes, followed by all FP32 scales.
    flat = kv_cache.view(pages, page_size * (dim + 4))
    scales = flat[:, page_size * dim :].view(torch.float32)
    q = q if q.dtype == torch.float16 else _bytes(q)
    out = torch.empty(
        (batch * next_n, max_model_len), dtype=torch.float32, device=q.device
    )
    if batch and next_n and max_model_len:
        _paged_logits[(batch * next_n, triton.cdiv(max_model_len, 64))](
            q,
            flat,
            scales,
            weights,
            context_lens,
            block_tables,
            out,
            *q.stride(),
            flat.stride(0),
            scales.stride(0),
            *weights.stride(),
            *context_lens.stride(),
            *block_tables.stride(),
            next_n,
            heads,
            dim,
            page_size,
            pages,
            block_tables.shape[1],
            max_model_len,
            max(16, triton.next_power_of_2(heads)),
            128,
            64,
            q.dtype == torch.float16,
            num_warps=4,
            num_stages=1,
        )
    return out
