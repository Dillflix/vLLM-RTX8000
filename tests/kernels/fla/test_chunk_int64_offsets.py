# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLA chunk addressing and RTX8000 GDN numerical regressions."""

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a GPU"
)

BT = 64


def _skip_if_low_memory(need: int) -> None:
    free, _ = current_platform.mem_get_info()
    if free < need + (1 << 30):
        pytest.skip("not enough free device memory")


@torch.inference_mode()
def test_chunk_delta_h_int64_seq_offset():
    """A later sequence's start offsets need int64.

    The second sequence starts at token 2^31 // (H * V), so its v, w and v_new
    offsets (bos * H * V) and its h offset (chunk * H * V * K) pass 2^31.
    """
    H, Hg, K, V = 64, 16, 128, 128
    L1, L2 = 2**31 // (H * V), 2 * BT
    T = L1 + L2
    # k, w, u, v_new and h, all bf16.
    _skip_if_low_memory(T * (Hg * K + 3 * H * V) * 2 + T // BT * H * V * K * 2)

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, device=device, generator=gen, dtype=torch.bfloat16)
        return x.mul_(scale)

    k = randn(1, T, Hg, K, scale=0.09)
    w = randn(1, T, H, K, scale=0.05)
    u = randn(1, T, H, V)
    # Chunk-local cumulative log decay, as chunk_local_cumsum produces it.
    g = torch.rand(1, T // BT, BT, H, device=device, generator=gen) * -0.05
    g = g.cumsum(2).view(1, T, H)
    h0 = torch.randn(2, H, V, K, device=device, generator=gen) * 0.1
    cu_seqlens = torch.tensor([0, L1, T], dtype=torch.int32, device=device)

    h, v_new, ht = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    # Reference: the second sequence on its own, where its offsets are 0.
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_fwd_h(
        k[:, L1:].contiguous(),
        w[:, L1:].contiguous(),
        u[:, L1:].contiguous(),
        g=g[:, L1:].contiguous(),
        initial_state=h0[1:].contiguous(),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, L2], dtype=torch.int32, device=device),
    )
    assert v_new is not None and ht is not None
    assert v_new_ref is not None and ht_ref is not None
    torch.testing.assert_close(h[:, L1 // BT :], h_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_new[:, L1:], v_new_ref, atol=0, rtol=0)
    torch.testing.assert_close(ht[1:], ht_ref, atol=0, rtol=0)


@torch.inference_mode()
def test_solve_tril_int64_offsets():
    """Rows past 2^31 // (H * BT) need int64 offsets.

    The first sequence runs past row 2^31 // (H * BT), which the per-row loads
    reach, and the second sequence starts past it, which the base offset
    (bos * H * BT) reaches.
    """
    H = 64
    L1, L2 = 2**31 // (H * BT) + 2 * BT, 2 * BT
    T = L1 + L2
    # A in fp32 and its inverse in bf16.
    _skip_if_low_memory(T * H * BT * (4 + 2))

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    # Strictly lower triangular inside each chunk of BT rows.
    A = torch.randn(1, T, H, BT, device=device, generator=gen).mul_(0.1)
    mask = torch.ones(BT, BT, device=device).tril(-1)
    A.view(1, T // BT, BT, H, BT).mul_(mask[:, None, :])
    cu_seqlens = torch.tensor([0, L1, T], dtype=torch.int32, device=device)

    Ai = solve_tril(A, cu_seqlens=cu_seqlens, output_dtype=torch.bfloat16)

    # Chunks are independent, so the last two chunks of each sequence on
    # their own are the reference.
    for end in (L1, T):
        start = end - 2 * BT
        ref = solve_tril(
            A[:, start:end].contiguous(),
            cu_seqlens=torch.tensor([0, 2 * BT], dtype=torch.int32, device=device),
            output_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(Ai[:, start:end], ref, atol=0, rtol=0)


def _gdn_inputs(lengths, index_dtype=torch.int32):
    """Independent requests, including chunk tails, in packed FLA layout."""
    torch.manual_seed(4319 + sum(lengths))
    total = sum(lengths)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, total, 8, 128, device="cuda"), dim=-1
        ).half()
        for _ in range(2)
    ]
    u = torch.randn(1, total, 24, 128, device="cuda").mul_(0.1).half()
    w = k.repeat_interleave(3, dim=2).mul(0.3).contiguous()
    g = torch.empty(1, total, 24, device="cuda")
    seq, indices, offsets = [0], [], [0]
    for n, length in enumerate(lengths):
        for chunk, start in enumerate(range(0, length, BT)):
            count = min(BT, length - start)
            increments = -torch.rand(count, 24, device="cuda") * 0.08 - 0.01
            g[0, seq[-1] + start : seq[-1] + start + count] = increments.cumsum(0)
            indices.append((n, chunk))
        seq.append(seq[-1] + length)
        offsets.append(len(indices))
    h0 = torch.randn(len(lengths), 24, 128, 128, device="cuda") * 0.05
    return dict(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        h0=h0,
        lengths=lengths,
        seq=torch.tensor(seq, dtype=index_dtype, device="cuda"),
        indices=torch.tensor(indices, dtype=index_dtype, device="cuda"),
        offsets=torch.tensor(offsets, dtype=index_dtype, device="cuda"),
    )


def _gdn_h_reference(data):
    """Chunk WY recurrence, with the documented intermediate FP16 roundings."""
    snapshots, finals = [], []
    updated = torch.full_like(data["u"], float("nan"))
    bos = 0
    for n, length in enumerate(data["lengths"]):
        state = data["h0"][n].clone()
        for start in range(bos, bos + length, BT):
            end = min(start + BT, bos + length)
            snapshots.append(state.half())
            key = data["k"][0, start:end].repeat_interleave(3, dim=1)
            key = key.transpose(0, 1).float()
            w = data["w"][0, start:end].transpose(0, 1).float()
            # The source kernel sums two K64 dots with FP16 state operands.
            half_state = state.half().float()
            residual = data["u"][0, start:end].transpose(0, 1).float()
            residual = residual - (
                w[..., :64] @ half_state[..., :64].transpose(-1, -2)
                + w[..., 64:] @ half_state[..., 64:].transpose(-1, -2)
            )
            updated[0, start:end] = residual.transpose(0, 1).half()
            gate = data["g"][0, start:end].T
            weighted = residual * (gate[:, -1:] - gate).exp().unsqueeze(-1)
            weighted = weighted.half().float()
            state = state * gate[:, -1].exp()[:, None, None]
            state = state + weighted.transpose(-1, -2) @ key
        finals.append(state)
        bos += length
    return torch.stack(snapshots)[None], updated, torch.stack(finals)


def _gdn_o_reference(data, h, v):
    """QH decay plus causal QK/PV; no FLA or native kernel in the reference."""
    out = torch.full_like(v, float("nan"))
    bos, chunk_id = 0, 0
    for length in data["lengths"]:
        for start in range(bos, bos + length, BT):
            end = min(start + BT, bos + length)
            q, k = [
                data[name][0, start:end]
                .repeat_interleave(3, dim=1)
                .transpose(0, 1)
                .float()
                for name in ("q", "k")
            ]
            gate = data["g"][0, start:end].T
            state = h[0, chunk_id].float()
            history = (q @ state.transpose(-1, -2)) * gate.exp().unsqueeze(-1)
            causal = torch.ones(end - start, end - start, device="cuda").tril()
            scores = (q @ k.transpose(-1, -2)) * (
                gate[:, :, None] - gate[:, None, :]
            ).exp()
            # FLA rounds the decayed QK scores to FP16 before the PV product.
            scores = (scores * causal).half().float()
            values = v[0, start:end].transpose(0, 1).float()
            result = (history + scores @ values) * 128**-0.5
            out[0, start:end] = result.transpose(0, 1).half()
            chunk_id += 1
        bos += length
    return out


def _gdn_h(data):
    return chunk_gated_delta_rule_fwd_h(
        data["k"],
        data["w"],
        data["u"],
        g=data["g"],
        initial_state=data["h0"],
        output_final_state=True,
        cu_seqlens=data["seq"],
        chunk_indices=data["indices"],
        chunk_offsets=data["offsets"],
    )


def _gdn_o(data, h, v, out):
    return chunk_fwd_o(
        data["q"],
        data["k"],
        v,
        h,
        g=data["g"],
        cu_seqlens=data["seq"],
        chunk_indices=data["indices"],
        core_attn_out=out.view(-1),
    )


def _assert_native_gdn_close(actual, reference):
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
    reference_rms = reference.float().square().mean().sqrt().item()
    error_rms = (actual.float() - reference.float()).square().mean().sqrt().item()
    relative_rms = error_rms / max(reference_rms, 1e-12)
    assert relative_rms <= 1e-3, (
        f"relative RMS={relative_rms:.6g}, error RMS={error_rms:.6g}, "
        f"reference RMS={reference_rms:.6g}, "
        f"reference max={reference.abs().max().item():.6g}"
    )


@pytest.mark.parametrize(
    "lengths", [(64,), (512,), (1024,), (2048,), (1,) * 5, (17, 63, 64, 65, 319)]
)
@torch.inference_mode()
def test_rtx8000_gdn_tile_dispatch_preserves_state_and_output(lengths, monkeypatch):
    """Tiles must preserve all outputs; unsupported shapes keep the real fallback."""
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN", "0")
    monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN_H", "0")
    monkeypatch.setenv("VLLM_RTX8000_GDN_TILES", "0")
    data = _gdn_inputs(lengths)
    original_state = data["h0"].clone()
    baseline = _gdn_h(data)
    out = torch.full_like(data["u"], float("nan"))
    expected = _gdn_o(data, baseline[0], baseline[1], out).clone()
    independent = _gdn_h_reference(data)
    for actual, reference in zip(baseline, independent):
        torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        expected,
        _gdn_o_reference(data, baseline[0], baseline[1]),
        atol=2e-3,
        rtol=2e-3,
    )

    monkeypatch.setenv("VLLM_RTX8000_GDN_TILES", "1")
    candidate = _gdn_h(data)
    out.fill_(float("nan"))
    actual = _gdn_o(data, candidate[0], candidate[1], out)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isfinite(actual).all()
    for actual_state, expected_state in zip(candidate, baseline):
        torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(data["h0"], original_state, atol=0, rtol=0)


@pytest.mark.skipif(
    current_platform.get_device_capability() != (7, 5),
    reason="Native GDN output requires SM75",
)
@pytest.mark.parametrize(
    "lengths,index_dtype",
    [
        ((64,), torch.int32),
        ((512,), torch.int32),
        ((1024,), torch.int32),
        ((2048,), torch.int32),
        ((1,) * 5, torch.int64),
        ((17, 63, 64, 65, 319), torch.int32),
    ],
)
@torch.inference_mode()
def test_rtx8000_native_gdn_output_and_graph(lengths, index_dtype, monkeypatch):
    """Poisoned outputs, packed tails and replay with changed inputs are written."""
    from vllm.models.qwen4_exp.nvidia.ops.rtx8000 import load_native

    load_native()
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_RTX8000_GDN_TILES", "0")
    monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN", "0")
    data = _gdn_inputs(lengths, index_dtype)
    chunks = len(data["indices"])
    h = torch.randn(1, chunks, 24, 128, 128, device="cuda").mul_(0.05).half()
    v = data["u"]
    expected = _gdn_o_reference(data, h, v)
    out = torch.full_like(v, float("nan"))
    # Native int64 addressing is checked against the independent formula.
    # Values fit int32 here; avoid autotuning a redundant Triton specialization.
    baseline_data = dict(data, seq=data["seq"].int(), indices=data["indices"].int())
    baseline = _gdn_o(baseline_data, h, v, out).clone()
    torch.testing.assert_close(baseline, expected, atol=2e-3, rtol=2e-3)

    def native(variant=0):
        torch.ops._rtx8000_C.gdn_o(
            data["q"],
            data["k"],
            v,
            h,
            data["g"],
            data["seq"],
            data["indices"],
            out,
            128**-0.5,
            variant,
        )

    for variant in (0, 1):
        out.fill_(float("nan"))
        native(variant)
        assert torch.isfinite(out).all()
        _assert_native_gdn_close(out, expected)
        torch.testing.assert_close(out, baseline, atol=2e-3, rtol=2e-3)
    monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN", "1")
    out.fill_(float("nan"))
    actual = _gdn_o(data, h, v, out)
    assert actual.data_ptr() == out.data_ptr()
    _assert_native_gdn_close(actual, expected)

    graph = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        native()
    # Same addresses, different contents: replay must not retain old results.
    v.mul_(0.5)
    h.mul_(0.25)
    data["q"].mul_(-0.5)
    expected = _gdn_o_reference(data, h, v)
    out.fill_(float("nan"))
    graph.replay()
    assert torch.isfinite(out).all()
    _assert_native_gdn_close(out, expected)


@pytest.mark.skipif(
    current_platform.get_device_capability() != (7, 5),
    reason="Native GDN state requires SM75",
)
@pytest.mark.parametrize(
    "lengths,index_dtype,offset_dtype",
    [
        ((64,), torch.int32, torch.int32),
        ((512,), torch.int32, torch.int32),
        ((1024,), torch.int32, torch.int32),
        ((2048,), torch.int32, torch.int32),
        ((8192,), torch.int32, torch.int32),
        ((1,) * 5, torch.int64, torch.int64),
        ((17, 63, 64, 65, 319), torch.int32, torch.int32),
        ((0, 17, 0, 65, 0), torch.int64, torch.int64),
        ((17, 63, 64, 65, 319), torch.int32, torch.int64),
    ],
)
@torch.inference_mode()
def test_rtx8000_native_gdn_state_and_graph(
    lengths, index_dtype, offset_dtype, monkeypatch
):
    """FP32 recurrence, half boundaries and live packed metadata survive replay."""
    from vllm.models.qwen4_exp.nvidia.ops.rtx8000 import load_native
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_offsets

    load_native()
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN_H", "0")
    monkeypatch.setenv("VLLM_RTX8000_GDN_TILES", "0")
    data = _gdn_inputs(lengths, index_dtype)
    if offset_dtype != index_dtype:
        # Real GDNAttentionMetadata builds these on CPU; cumsum promotes to INT64.
        data["offsets"] = prepare_chunk_offsets(data["seq"].cpu(), BT).to("cuda")
        assert data["offsets"].dtype == offset_dtype
    original_state = data["h0"].clone()
    independent = _gdn_h_reference(data)
    # These metadata values fit INT32. Check native INT64 against the independent
    # equations without compiling a duplicate Triton pointer specialization.
    baseline = _gdn_h(
        dict(
            data,
            seq=data["seq"].int(),
            indices=data["indices"].int(),
            offsets=data["offsets"].int(),
        )
    )
    for actual, expected in zip(baseline, independent):
        _assert_native_gdn_close(actual, expected)

    for variant in (0, 1):
        outputs = tuple(torch.full_like(t, float("nan")) for t in baseline)

        def native(outputs=outputs, variant=variant):
            torch.ops._rtx8000_C.gdn_h(
                data["k"],
                data["w"],
                data["u"],
                data["g"],
                data["h0"],
                data["seq"],
                data["offsets"],
                *outputs,
                variant,
            )

        native()
        for actual, expected, triton_result in zip(outputs, independent, baseline):
            assert torch.isfinite(actual).all()
            _assert_native_gdn_close(actual, expected)
            _assert_native_gdn_close(actual, triton_result)
        torch.testing.assert_close(data["h0"], original_state, atol=0, rtol=0)
        if variant == 1:
            monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN_H", "1")
            dispatched = _gdn_h(data)
            for actual, expected in zip(dispatched, outputs):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            monkeypatch.setenv("VLLM_RTX8000_NATIVE_GDN_H", "0")

        graph = torch.cuda.CUDAGraph()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            native()
        saved = {
            name: tensor.clone()
            for name, tensor in data.items()
            if isinstance(tensor, torch.Tensor)
        }
        changed_lengths = tuple(reversed(lengths))
        changed = _gdn_inputs(changed_lengths, index_dtype)
        data["lengths"] = changed_lengths
        for name in ("seq", "offsets", "g"):
            data[name].copy_(changed[name])
        data["u"].mul_(0.5)
        data["w"].mul_(0.75)
        data["h0"].mul_(0.25)
        changed_reference = _gdn_h_reference(data)
        for tensor in outputs:
            tensor.fill_(float("nan"))
        graph.replay()
        for actual, expected in zip(outputs, changed_reference):
            assert torch.isfinite(actual).all()
            _assert_native_gdn_close(actual, expected)
        for name, tensor in saved.items():
            data[name].copy_(tensor)
        data["lengths"] = lengths
