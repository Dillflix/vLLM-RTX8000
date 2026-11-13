# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FP16 mHC fusion, retaining the FP32 cuBLAS projection.

Selection requires the explicit glm_mhc_backend='turing' option. Four
residual streams are supported; projection weights and all mixing math stay
FP32. The contracted activation rounds to FP16 before optional RMSNorm, as in
the existing reference path.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _sum_four_streams(x):
    # Match torch.sum(dim=-2) for the four-stream mixing matrix. The row
    # reduction uses a tree; this strided column reduction is sequential.
    stream = tl.arange(0, 4)[:, None]
    result = tl.sum(tl.where(stream == 0, x, 0.0), axis=0)
    for i in tl.static_range(1, 4):
        result += tl.sum(tl.where(stream == i, x, 0.0), axis=0)
    return result


@triton.jit
def _pre_mix(
    X,
    PROJ,
    SCALE,
    BASE,
    INV,
    POST,
    COMB,
    OUT,
    H: tl.constexpr,
    B: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    REPEAT: tl.constexpr,
):
    row = tl.program_id(0)
    stream = tl.arange(0, 4)
    dim = tl.arange(0, B)
    inv = tl.load(INV + row)
    s0, s1, s2 = tl.load(SCALE), tl.load(SCALE + 1), tl.load(SCALE + 2)
    pre = tl.load(PROJ + row * 24 + stream) * inv
    pre_logits = pre * s0 + tl.load(BASE + stream)
    pre = tl.div_rn(1.0, 1.0 + tldevice.exp(-pre_logits)) + PRE_EPS
    post = tl.load(PROJ + row * 24 + 4 + stream) * inv
    post_logits = post * s1 + tl.load(BASE + 4 + stream)
    post = tl.div_rn(1.0, 1.0 + tldevice.exp(-post_logits)) * POST_MULT
    offsets = stream[:, None] * 4 + stream[None, :]
    comb = tl.load(PROJ + row * 24 + 8 + offsets) * inv
    comb = comb * s2 + tl.load(BASE + 8 + offsets)
    comb = tldevice.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = tl.div_rn(comb, tl.sum(comb, axis=1)[:, None]) + SINK_EPS
    comb = tl.div_rn(comb, _sum_four_streams(comb)[None, :] + SINK_EPS)
    for _ in range(REPEAT - 1):
        comb = tl.div_rn(comb, tl.sum(comb, axis=1)[:, None] + SINK_EPS)
        comb = tl.div_rn(comb, _sum_four_streams(comb)[None, :] + SINK_EPS)
    # Match torch's sequential four-stream reduction before FP16 rounding.
    y = tl.full((B,), 0.0, tl.float32)
    for i in tl.static_range(4):
        coefficient = tl.sum(tl.where(stream == i, pre, 0.0), axis=0)
        values = tl.load(X + row * 4 * H + i * H + dim, dim < H, 0.0).to(tl.float32)
        y += coefficient * values
    y = y.to(tl.float16)
    tl.store(POST + row * 4 + stream, post)
    tl.store(COMB + row * 16 + offsets, comb)
    tl.store(OUT + row * H + dim, y, dim < H)


@triton.jit
def _post_mix(
    X,
    RESIDUAL,
    POST,
    COMB,
    OUT,
    OUT_FLOAT,
    H: tl.constexpr,
    B: tl.constexpr,
    WRITE_FLOAT: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    dim = tile * B + tl.arange(0, B)
    x = tl.load(X + row * H + dim, dim < H, 0.0).to(tl.float32)
    r0 = tl.load(RESIDUAL + row * 4 * H + dim, dim < H, 0.0).to(tl.float32)
    r1 = tl.load(RESIDUAL + row * 4 * H + H + dim, dim < H, 0.0).to(tl.float32)
    r2 = tl.load(RESIDUAL + row * 4 * H + 2 * H + dim, dim < H, 0.0).to(tl.float32)
    r3 = tl.load(RESIDUAL + row * 4 * H + 3 * H + dim, dim < H, 0.0).to(tl.float32)
    for j in tl.static_range(4):
        value = r0 * tl.load(COMB + row * 16 + j)
        value = tl.fma(r1, tl.load(COMB + row * 16 + 4 + j), value)
        value = tl.fma(r2, tl.load(COMB + row * 16 + 8 + j), value)
        value = tl.fma(r3, tl.load(COMB + row * 16 + 12 + j), value)
        value += x * tl.load(POST + row * 4 + j)
        value = value.to(tl.float16)
        offset = row * 4 * H + j * H + dim
        tl.store(OUT + offset, value, dim < H)
        if WRITE_FLOAT:
            tl.store(OUT_FLOAT + offset, value.to(tl.float32), dim < H)


def _check_residual(residual):
    if residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("Turing mHC requires [tokens, 4, hidden] residuals")
    if residual.dtype != torch.float16 or not residual.is_contiguous():
        raise ValueError("Turing mHC requires contiguous FP16 residuals")


def _pre_from_float(
    x,
    mix_input,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_weight,
    norm_eps,
):
    tokens, width = x.shape
    hidden = width // 4
    if fn.shape != (24, width) or hc_scale.shape != (3,) or hc_base.shape != (24,):
        raise ValueError("Invalid four-stream mHC projection/gate geometry")
    if any(t.dtype != torch.float32 for t in (fn, hc_scale, hc_base)):
        raise ValueError("mHC projection and gate parameters must remain FP32")
    if norm_weight is not None and norm_weight.shape != (hidden,):
        raise ValueError("mHC norm weight must match hidden size")
    if sinkhorn_repeat < 1:
        raise ValueError("mHC needs at least one Sinkhorn iteration")
    post = torch.empty((tokens, 4, 1), dtype=torch.float32, device=x.device)
    comb = torch.empty((tokens, 4, 4), dtype=torch.float32, device=x.device)
    output = torch.empty((tokens, hidden), dtype=torch.float16, device=x.device)
    if tokens:
        # FP32 GEMM is unchanged from mhc_pre_torch; no FP16 projection shortcut.
        projection = torch.mm(x, fn.t())
        # x is private FP32 scratch. The original FP16 values remain available
        # in mix_input, so square in place after the projection consumes x.
        # Keep the reference reduction order at FP16 rounding boundaries.
        inv = torch.rsqrt(x.square_().sum(dim=-1) / width + rms_eps)
        _pre_mix[(tokens,)](
            mix_input,
            projection,
            hc_scale,
            hc_base,
            inv,
            post,
            comb,
            output,
            hidden,
            triton.next_power_of_2(hidden),
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            num_warps=8,
            enable_fp_fusion=False,
        )
    if norm_weight is not None:
        from vllm import ir

        output = ir.ops.rms_norm(output, norm_weight, norm_eps)
    return post, comb, output


def mhc_pre_turing(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    *,
    norm_weight=None,
    norm_eps=0.0,
):
    """Experimental four-stream pre-op with FP32 projection and mixing."""
    _check_residual(residual)
    x = residual.flatten(1).float()
    return _pre_from_float(
        x,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )


def _post_float_workspace(residual, live_inputs, *, max_tokens=0):
    """Borrow only until fused post-pre returns; returned outputs never alias it."""
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("mHC post workspace token limit must be a nonnegative integer")
    if max_tokens == 0 or residual.shape[0] == 0:
        return None
    _check_residual(residual)
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import current_workspace_manager

    tokens, _, hidden = residual.shape
    if (
        not current_platform.is_device_capability(75)
        or not residual.is_cuda
        or hidden != 4096
        or tokens > max_tokens
        or any(
            value is not None and value.device != residual.device
            for value in live_inputs
        )
    ):
        raise ValueError("mHC post workspace requires bounded SM75 FP16 H4096 inputs")
    manager = current_workspace_manager()
    if manager._num_ubatches != 1 or manager._num_lanes != 1:
        raise ValueError("mHC post workspace requires one execution lane/ubatch")
    (scratch,) = manager.get_simultaneous(((tokens, 4 * hidden), torch.float32))
    if scratch.device != residual.device:
        raise ValueError("mHC post workspace is on a different device")
    if any(
        value is not None and torch._C._overlaps(scratch, value)
        for value in live_inputs
    ):
        return None
    return scratch


def _post(x, residual, post_layer_mix, comb_res_mix, write_float, fp32_out=None):
    _check_residual(residual)
    tokens, _, hidden = residual.shape
    if x.shape != (tokens, hidden) or x.dtype != torch.float16 or not x.is_contiguous():
        raise ValueError("mHC layer output must be contiguous FP16 [tokens, hidden]")
    if post_layer_mix.shape != (tokens, 4, 1) or comb_res_mix.shape != (tokens, 4, 4):
        raise ValueError("Invalid mHC post-mixing geometry")
    for t in (post_layer_mix, comb_res_mix):
        if t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError("mHC mixing coefficients must be contiguous FP32")
    output = torch.empty_like(residual)
    if fp32_out is not None and (
        not write_float
        or fp32_out.shape != (tokens, 4 * hidden)
        or fp32_out.dtype != torch.float32
        or fp32_out.device != x.device
        or not fp32_out.is_contiguous()
        or any(
            torch._C._overlaps(fp32_out, value)
            for value in (x, residual, post_layer_mix, comb_res_mix, output)
        )
    ):
        raise ValueError("Invalid or aliased mHC FP32 post workspace")
    fp32 = (
        fp32_out
        if fp32_out is not None
        else (
            torch.empty((tokens, 4 * hidden), dtype=torch.float32, device=x.device)
            if write_float
            else None
        )
    )
    if tokens:
        _post_mix[(tokens, triton.cdiv(hidden, 256))](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            output,
            fp32,
            hidden,
            256,
            write_float,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, fp32


def mhc_post_turing(x, residual, post_layer_mix, comb_res_mix):
    """Experimental FP32 post-mixing with one FP16 output rounding."""
    return _post(x, residual, post_layer_mix, comb_res_mix, False)[0]


def mhc_fused_post_pre_turing(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    *,
    norm_weight=None,
    norm_eps=0.0,
    post_workspace_tokens=0,
):
    """Fuse post-mixing with the FP32 projection input's materialization."""
    scratch = _post_float_workspace(
        residual,
        (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, norm_weight),
        max_tokens=post_workspace_tokens,
    )
    current, fp32 = _post(
        x, residual, post_layer_mix, comb_res_mix, True, fp32_out=scratch
    )
    post, comb, output = _pre_from_float(
        fp32,
        current,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )
    return current, post, comb, output
