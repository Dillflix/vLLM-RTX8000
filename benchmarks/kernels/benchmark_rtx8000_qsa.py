# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native SM75 Tensor Core QSA with the production sparse kernel."""

import argparse
import itertools
import json
from functools import partial as bind
from pathlib import Path

import torch
from benchmark_rtx8000_dense import operating_snapshot, time_calls

from vllm.models.qwen4_exp.nvidia import model as model_registration  # noqa: F401
from vllm.models.qwen4_exp.nvidia.ops import qsa
from vllm.models.qwen4_exp.nvidia.ops.rtx8000 import load_native
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 5, 20, 512, 1024, 2048]
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[32768, 65536, 131072]
    )
    parser.add_argument("--variants", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--splits", type=int, nargs="+")
    parser.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(5678)
    torch.set_num_threads(2)
    load_native()
    report = {
        "rows": [],
        "failures": [],
        "complete": False,
        "operating_start": operating_snapshot(),
    }
    flush = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    # Fixed 128K allocation; test sparse indices drawn from 32K, 64K, 128K.
    packed = torch.randn((83, 1, 1600, 512), device="cuda", dtype=torch.float16)
    k, v = packed.transpose(1, 2).split(256, dim=-1)
    table = torch.randperm(83, device="cuda", dtype=torch.int32)[None]
    for context in args.contexts:
        for M in args.tokens:
            query = torch.randn(M, 12, 256, device="cuda", dtype=torch.float16)
            gate = torch.randn_like(query)
            idx = torch.randint(0, context, (M, 2052), device="cuda", dtype=torch.int32)
            idx[:, -1] = 2051
            requests = torch.zeros(M, device="cuda", dtype=torch.int32)
            ref = torch.empty_like(query)
            baseline = bind(
                qsa._qsa_sparse_paged_splitk,
                query,
                k,
                v,
                idx,
                table,
                requests,
                M >= 512,
                ref,
                gate,
                1 / 16,
                1.0,
            )
            baseline()
            for row in range(min(M, 2)):
                logical = idx[row, :-1].long()
                physical = table[0, logical // 1600].long()
                keys = k[physical, logical % 1600, 0].float()
                values = v[physical, logical % 1600, 0].float()
                expected = (
                    (query[row].float() @ keys.T / 16).softmax(-1) @ values
                ).half().float() * gate[row].float().sigmoid()
                torch.testing.assert_close(
                    ref[row].float(), expected, atol=2e-3, rtol=2e-3
                )
            baseline_times = {
                mode: time_calls(baseline, flush, True, mode, args.samples, 1)
                for mode in args.modes
            }
            plans = list(
                itertools.product(
                    args.variants,
                    args.splits or ((1, 4, 16, 32) if M < 512 else (1, 4)),
                )
            )
            # Reverse order each round to expose clock/cache/order effects.
            cases = [
                (repeat, variant, splits)
                for repeat in range(args.repeats)
                for variant, splits in (plans if repeat % 2 == 0 else plans[::-1])
            ]
            for repeat, variant, splits in cases:
                out = torch.empty_like(query)
                partial = torch.empty(
                    splits, M, 12, 256, device="cuda", dtype=torch.float32
                )
                lse = torch.empty(splits, M, 12, device="cuda", dtype=torch.float32)

                def run(
                    query=query,
                    idx=idx,
                    requests=requests,
                    gate=gate,
                    out=out,
                    partial=partial,
                    lse=lse,
                    splits=splits,
                    M=M,
                    variant=variant,
                ):
                    torch.ops._rtx8000_C.qsa(
                        query,
                        k,
                        v,
                        idx,
                        table,
                        requests,
                        gate,
                        out,
                        partial,
                        lse,
                        splits,
                        variant,
                    )
                    if splits > 1:
                        qsa._qsa_merge_splitk_kernel[(M, 12)](
                            partial,
                            lse,
                            out,
                            gate,
                            out.stride(0),
                            out.stride(1),
                            gate.stride(0),
                            gate.stride(1),
                            M,
                            HEAD_DIM=256,
                            NUM_QUERY_HEADS=12,
                            NUM_SPLITS=splits,
                            BLOCK_SPLITS=triton.next_power_of_2(splits),
                            num_warps=2,
                            num_stages=1,
                        )

                try:
                    run()
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
                    for mode in args.modes:
                        timing = time_calls(run, flush, True, mode, args.samples, 1)
                        row = {
                            "context": context,
                            "M": M,
                            "splits": splits,
                            "variant": variant,
                            "repeat": repeat,
                            "mode": mode,
                            "cold_cache": True,
                            "baseline_ms": baseline_times[mode]["median_ms"],
                            "native_ms": timing["median_ms"],
                            "samples_ms": timing["samples_ms"],
                            "speedup": baseline_times[mode]["median_ms"]
                            / timing["median_ms"],
                            "max_abs_error": (out - ref).abs().max().item(),
                        }
                        report["rows"].append(row)
                        print(json.dumps(row), flush=True)
                except Exception as error:
                    row = {
                        "context": context,
                        "M": M,
                        "splits": splits,
                        "variant": variant,
                        "error": str(error),
                    }
                    report["failures"].append(row)
                args.output.write_text(json.dumps(report, indent=2))
    report["complete"] = True
    report["operating_end"] = operating_snapshot()
    args.output.write_text(json.dumps(report, indent=2))
    if report["failures"]:
        raise RuntimeError(f"{len(report['failures'])} QSA cases failed")


if __name__ == "__main__":
    main()
