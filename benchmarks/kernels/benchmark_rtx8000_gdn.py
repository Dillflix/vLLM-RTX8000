# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify RTX8000 GDN H/O kernels and time the actual FLA forward chain.

Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        benchmarks/kernels/benchmark_rtx8000_gdn.py --output gdn.json

Lengths are TOTAL scheduled tokens, divided among the requested sequences.
The default includes chunk tails and one-token requests, which must retain
correct fallback behavior. These are kernel measurements, not serving token/s.
The existing T64 startup autotuning runs once in this process, outside timing.
Both eager and CUDA graph timings are reported; graph results do not imply
that the serving engine currently captures the prefill chain.

"""

import argparse
import contextlib
import datetime
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from vllm import envs
from vllm.models.qwen4_exp.nvidia.ops.rtx8000 import load_native
from vllm.third_party.flash_linear_attention.ops import chunk_delta_h, chunk_o
from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_offsets
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd


@contextlib.contextmanager
def mode(name):
    settings = {
        "VLLM_RTX8000_GDN_TILES": str(
            int(name in ("tiles", "tiles_native", "state_native", "all_native"))
        ),
        "VLLM_RTX8000_NATIVE_GDN": str(
            int(name in ("native", "tiles_native", "all_native"))
        ),
        "VLLM_RTX8000_NATIVE_GDN_H": str(int(name in ("state_native", "all_native"))),
    }
    before = {key: os.environ.get(key) for key in settings}
    envs.disable_envs_cache()
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_inputs(lengths):
    torch.manual_seed(4319 + sum(lengths))
    total = sum(lengths)
    q, k = [
        torch.nn.functional.normalize(
            torch.randn(1, total, 8, 128, device="cuda"), dim=-1
        ).half()
        for _ in range(2)
    ]
    v = torch.randn(1, total, 24, 128, device="cuda").mul_(0.1).half()
    g = -torch.rand(1, total, 24, device="cuda") * 0.08 - 0.01
    # fused_post_conv_prep emits FP32 g and beta on the real model path.
    beta = torch.rand(1, total, 24, device="cuda") * 0.6 + 0.1
    h0 = torch.randn(len(lengths), 24, 128, 128, device="cuda") * 0.05
    seq, indices = [0], []
    for n, length in enumerate(lengths):
        indices.extend((n, chunk) for chunk in range((length + 63) // 64))
        seq.append(seq[-1] + length)
    return dict(
        q=q,
        k=k,
        v=v,
        log_gate=g,
        beta=beta,
        h0=h0,
        lengths=lengths,
        seq=torch.tensor(seq, dtype=torch.int32, device="cuda"),
        indices=torch.tensor(indices, dtype=torch.int32, device="cuda"),
        offsets=prepare_chunk_offsets(torch.tensor(seq, dtype=torch.int32), 64).to(
            "cuda"
        ),
    )


def metadata(data):
    return dict(cu_seqlens=data["seq"], chunk_indices=data["indices"])


def prepare_wy(data):
    """Use the real chain's WY operands, not an approximation to W and U."""
    meta = metadata(data)
    g = chunk_local_cumsum(data["log_gate"], chunk_size=64, **meta)
    a = chunk_scaled_dot_kkt_fwd(
        k=data["k"], beta=data["beta"], g=g, output_dtype=torch.float32, **meta
    )
    a = solve_tril(a, output_dtype=torch.float16, **meta)
    w, u = recompute_w_u_fwd(
        k=data["k"], v=data["v"], beta=data["beta"], A=a, g_cumsum=g, **meta
    )
    return dict(data, g=g, w=w, u=u)


def run_h(data):
    return chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=data["k"],
        w=data["w"],
        u=data["u"],
        g=data["g"],
        initial_state=data["h0"],
        output_final_state=True,
        chunk_offsets=data["offsets"],
        **metadata(data),
    )


def run_o(data, h, v, out):
    return chunk_o.chunk_fwd_o(
        q=data["q"],
        k=data["k"],
        v=v,
        h=h,
        g=data["g"],
        scale=128**-0.5,
        core_attn_out=out.view(-1),
        **metadata(data),
    )


def run_chain(data, out):
    return chunk_gated_delta_rule(
        q=data["q"],
        k=data["k"],
        v=data["v"],
        g=data["log_gate"],
        beta=data["beta"],
        initial_state=data["h0"],
        output_final_state=True,
        chunk_offsets=data["offsets"],
        core_attn_out=out.view(-1),
        **metadata(data),
    )


def output_reference(data, h, v):
    """Independent FP32 products, including causal decay and FP16 A rounding."""
    out = torch.full_like(v, float("nan"))
    bos, chunk_id = 0, 0
    for length in data["lengths"]:
        for start in range(bos, bos + length, 64):
            end = min(start + 64, bos + length)
            q, k = [
                data[key][0, start:end]
                .repeat_interleave(3, dim=1)
                .transpose(0, 1)
                .float()
                for key in ("q", "k")
            ]
            gate = data["g"][0, start:end].T
            history = q @ h[0, chunk_id].float().transpose(-1, -2)
            history *= gate.exp().unsqueeze(-1)
            scores = (q @ k.transpose(-1, -2)) * (
                gate[:, :, None] - gate[:, None, :]
            ).exp()
            mask = torch.ones(end - start, end - start, device="cuda").tril()
            scores = (scores * mask).half().float()
            result = history + scores @ v[0, start:end].transpose(0, 1).float()
            out[0, start:end] = (result * 128**-0.5).transpose(0, 1).half()
            chunk_id += 1
        bos += length
    return out


def recurrent_reference(data):
    """Independent token-wise GDN equations before chunk/WY rounding."""
    out = torch.full_like(data["v"], float("nan"))
    final, bos = [], 0
    for n, length in enumerate(data["lengths"]):
        state = data["h0"][n].clone()
        for token in range(bos, bos + length):
            q, k = [
                data[name][0, token].repeat_interleave(3, dim=0).float()
                for name in ("q", "k")
            ]
            state *= data["log_gate"][0, token].exp()[:, None, None]
            prediction = (state * k[:, None, :]).sum(-1)
            residual = data["v"][0, token].float() - prediction
            residual *= data["beta"][0, token, :, None].float()
            state += residual[:, :, None] * k[:, None, :]
            out[0, token] = ((state * q[:, None, :]).sum(-1) * 128**-0.5).half()
        final.append(state)
        bos += length
    return out, torch.stack(final)


def state_reference(data):
    """Independent WY recurrence with FP32 state and FP16 operand boundaries."""
    snapshots, finals = [], []
    updated = torch.full_like(data["u"], float("nan"))
    bos = 0
    for n, length in enumerate(data["lengths"]):
        state = data["h0"][n].clone()
        for start in range(bos, bos + length, 64):
            end = min(start + 64, bos + length)
            snapshots.append(state.half())
            key = (
                data["k"][0, start:end]
                .repeat_interleave(3, dim=1)
                .transpose(0, 1)
                .float()
            )
            w = data["w"][0, start:end].transpose(0, 1).float()
            rounded = state.half().float()
            prediction = w[..., :64] @ rounded[..., :64].transpose(-1, -2) + w[
                ..., 64:
            ] @ rounded[..., 64:].transpose(-1, -2)
            residual = data["u"][0, start:end].transpose(0, 1).float() - prediction
            updated[0, start:end] = residual.transpose(0, 1).half()
            gate = data["g"][0, start:end].T
            weighted = (
                (residual * (gate[:, -1:] - gate).exp().unsqueeze(-1)).half().float()
            )
            state = state * gate[:, -1].exp()[:, None, None]
            state = state + weighted.transpose(-1, -2) @ key
        finals.append(state)
        bos += length
    return torch.stack(snapshots)[None], updated, torch.stack(finals)


def errors(actual, expected, exact=False, relative_rms_limit=None):
    torch.testing.assert_close(
        actual, expected, atol=0 if exact else 2e-3, rtol=0 if exact else 2e-3
    )
    delta = actual.float() - expected.float()
    rms = delta.square().mean().sqrt().item()
    reference_rms = expected.float().square().mean().sqrt().item()
    relative_rms = rms / max(reference_rms, 1e-12)
    if relative_rms_limit is not None:
        assert relative_rms <= relative_rms_limit, (
            f"relative RMS {relative_rms:.6g} exceeds {relative_rms_limit}; "
            f"error RMS={rms:.6g}, reference RMS={reference_rms:.6g}"
        )
    return dict(
        max_abs=delta.abs().max().item(),
        rms=rms,
        reference_rms=reference_rms,
        reference_abs_max=expected.abs().max().item(),
        relative_rms=relative_rms,
        finite=bool(torch.isfinite(actual).all()),
    )


def event_time(fn, samples, replays, graph=False):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    captured = None
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            for _ in range(replays):
                fn()
    times = []
    for _ in range(samples):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        if captured is None:
            for _ in range(replays):
                fn()
        else:
            captured.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / replays)
    return dict(median_ms=statistics.median(times), samples_ms=times)


def selected_config(kernel):
    config = getattr(kernel.fn, "best_config", None)
    if config is None:
        return None
    return dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 512, 1024, 2048])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--no-tail-cases",
        action="store_true",
        help="Time only requested shapes; use after separate tail correctness tests",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "baseline",
            "tiles",
            "native",
            "tiles_native",
            "state_native",
            "all_native",
        ],
        default=["baseline", "tiles", "native", "tiles_native", "all_native"],
    )
    parser.add_argument("--native-variants", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--state-native-variants", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This benchmark requires an SM75 GPU")
    if min(args.lengths + args.requests + [args.samples, args.replays]) < 1:
        parser.error("Lengths, request counts, samples and replays must be positive")
    torch.set_num_threads(2)
    report = dict(
        started=datetime.datetime.now(datetime.UTC).isoformat(),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        scope="H24/Hg8/K128/V128/BT64; real WY and full GDN; total scheduled tokens",
        timing="CUDA events; median plus raw samples; startup/compilation excluded",
        caveat="Synthetic inputs, not serving throughput or a model-quality result",
        rows=[],
        failures=[],
        complete=False,
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def emit(row):
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        save()

    # The real layer warms at T64. Do this once for all cases, not per variant.
    print("Priming the existing T64 full-chain autotuners once", flush=True)
    started = time.monotonic()
    with mode("baseline"):
        warm = make_inputs([64])
        run_chain(warm, torch.empty_like(warm["v"]))
    torch.accelerator.synchronize()
    report["startup_seconds"] = time.monotonic() - started
    report["baseline_h_config"] = selected_config(
        chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    )
    report["baseline_o_config"] = selected_config(chunk_o.chunk_fwd_kernel_o)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("startup_seconds", "baseline_h_config", "baseline_o_config")
            }
        ),
        flush=True,
    )
    save()
    if any("native" in name for name in args.modes):
        load_native()

    cases = []
    for total in args.lengths:
        for requests in args.requests:
            if total >= requests:
                q, r = divmod(total, requests)
                cases.append([q + (i < r) for i in range(requests)])
    if not args.no_tail_cases:
        cases.extend([[1] * 5, [17, 63, 64, 65, 319]])
    for lengths in dict.fromkeys(tuple(lengths) for lengths in cases):
        try:
            with mode("baseline"):
                data = prepare_wy(make_inputs(lengths))
                original_state = data["h0"].clone()
                base_h = run_h(data)
                scratch = torch.full_like(data["v"], float("nan"))
                base_o = run_o(data, base_h[0], base_h[1], scratch).clone()
                scratch.fill_(float("nan"))
                base_chain = tuple(x.clone() for x in run_chain(data, scratch))
            independent_o = output_reference(data, base_h[0], base_h[1])
            independent_chain = recurrent_reference(data)
            independent_h = state_reference(data)
            emit(
                dict(
                    lengths=lengths,
                    total_tokens=sum(lengths),
                    mode="reference",
                    o=errors(base_o, independent_o),
                    h=[
                        errors(x, y, relative_rms_limit=1e-3)
                        for x, y in zip(base_h, independent_h)
                    ],
                    chain_output=errors(base_chain[0], independent_chain[0]),
                    chain_final_state=errors(base_chain[1], independent_chain[1]),
                )
            )
            for name in args.modes:
                with mode(name):
                    native_rms_limit = 1e-3 if "native" in name else None
                    native_state = name in ("state_native", "all_native")
                    h = run_h(data)
                    scratch.fill_(float("nan"))
                    actual_o = run_o(data, h[0], h[1], scratch).clone()
                    scratch.fill_(float("nan"))
                    chain = run_chain(data, scratch)
                    numerical = dict(
                        h=[
                            errors(
                                x,
                                y,
                                exact=not native_state,
                                relative_rms_limit=1e-3 if native_state else None,
                            )
                            for x, y in zip(h, base_h)
                        ],
                        independent_h=[
                            errors(x, y, relative_rms_limit=1e-3)
                            for x, y in zip(h, independent_h)
                        ],
                        o=errors(
                            actual_o,
                            base_o,
                            exact="native" not in name,
                            relative_rms_limit=native_rms_limit,
                        ),
                        independent_o=errors(
                            actual_o, independent_o, relative_rms_limit=native_rms_limit
                        ),
                        chain_output=errors(
                            chain[0],
                            base_chain[0],
                            exact="native" not in name,
                            relative_rms_limit=native_rms_limit,
                        ),
                        chain_final_state=errors(
                            chain[1],
                            base_chain[1],
                            exact=not native_state,
                            relative_rms_limit=1e-3 if native_state else None,
                        ),
                        initial_state=errors(data["h0"], original_state, exact=True),
                    )
                    for scope, fn in (
                        ("h", partial(run_h, data)),
                        ("o", partial(run_o, data, h[0], h[1], scratch)),
                        ("full_chain", partial(run_chain, data, scratch)),
                    ):
                        emit(
                            dict(
                                lengths=lengths,
                                total_tokens=sum(lengths),
                                mode=name,
                                scope=scope,
                                numerical=numerical,
                                eager=event_time(fn, args.samples, args.replays),
                                graph=event_time(
                                    fn, args.samples, args.replays, graph=True
                                ),
                            )
                        )
            if any(name in ("state_native", "all_native") for name in args.modes):
                for variant in args.state_native_variants:
                    state_outputs = tuple(
                        torch.full_like(t, float("nan")) for t in base_h
                    )
                    native_h = partial(
                        torch.ops._rtx8000_C.gdn_h,
                        data["k"],
                        data["w"],
                        data["u"],
                        data["g"],
                        data["h0"],
                        data["seq"],
                        data["offsets"],
                        *state_outputs,
                        variant,
                    )
                    native_h()
                    numerical = [
                        errors(x, y, relative_rms_limit=1e-3)
                        for x, y in zip(state_outputs, independent_h)
                    ]
                    emit(
                        dict(
                            lengths=lengths,
                            total_tokens=sum(lengths),
                            mode="direct_native",
                            scope="h",
                            variant=variant,
                            numerical=numerical,
                            eager=event_time(native_h, args.samples, args.replays),
                            graph=event_time(
                                native_h, args.samples, args.replays, graph=True
                            ),
                        )
                    )
            if any(
                name in ("native", "tiles_native", "all_native") for name in args.modes
            ):
                for variant in args.native_variants:
                    native = partial(
                        torch.ops._rtx8000_C.gdn_o,
                        data["q"],
                        data["k"],
                        base_h[1],
                        base_h[0],
                        data["g"],
                        data["seq"],
                        data["indices"],
                        scratch,
                        128**-0.5,
                        variant,
                    )

                    scratch.fill_(float("nan"))
                    native()
                    numerical = errors(scratch, independent_o, relative_rms_limit=1e-3)
                    emit(
                        dict(
                            lengths=lengths,
                            total_tokens=sum(lengths),
                            mode="direct_native",
                            scope="o",
                            variant=variant,
                            numerical=numerical,
                            eager=event_time(native, args.samples, args.replays),
                            graph=event_time(
                                native, args.samples, args.replays, graph=True
                            ),
                        )
                    )
        except Exception as error:
            failure = dict(lengths=lengths, error=repr(error))
            report["failures"].append(failure)
            print(json.dumps(failure), flush=True)
            save()
            raise
    report["complete"] = True
    report["finished"] = datetime.datetime.now(datetime.UTC).isoformat()
    save()


if __name__ == "__main__":
    main()
