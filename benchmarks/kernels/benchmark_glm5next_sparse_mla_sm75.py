# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic GLM TP2 NoPE-512 sparse MLA profile, not serving throughput.

Measures physical index conversion plus attention on contiguous/interleaved
paged caches. All candidates pass an independent FP32 softmax reference before
timing. Graph replay timings do not imply full-model graph support. The existing
autotuner and production defaults are unchanged; query sizes run in listed order.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from benchmark_rtx8000_marlin import measure

from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops import triton_mla_sparse_sm75 as mla


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 5, 512, 2048])
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 4, 8, 16])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--interleaves", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--kernel-module", type=Path, help="Isolated candidate Python file"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.context % 64 or args.context < 2176:
        parser.error("Context must be a multiple of64 and at least2176")
    if min(*args.tokens, args.iterations, args.repeats) < 1:
        parser.error("Query counts and repeat counts must be positive")
    if any(s not in (0, 1, 2, 4, 8, 16) for s in args.splits):
        parser.error("Split choices must be0(auto),1,2,4,8,16")
    if any(i not in (1, 2) for i in args.interleaves):
        parser.error("Interleave choices must be1 or2")
    kernel_module = mla
    if args.kernel_module:
        spec = importlib.util.spec_from_file_location(
            "mla_candidate", args.kernel_module
        )
        kernel_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernel_module
        spec.loader.exec_module(kernel_module)
    torch.accelerator.set_device_index(0)
    if torch.cuda.get_device_capability() != (7, 5):
        parser.error("This profile requires SM75")
    torch.manual_seed(127)
    torch.set_num_threads(2)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    report = {
        "scope": "synthetic physical-index conversion + one sparse MLA layer",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {"heads": 32, "qk": 512, "value": 512, "padded_topk": 2176},
        "valid_keys": 2051,
        "parameters": vars(args)
        | {
            "output": str(args.output),
            "kernel_module": str(args.kernel_module) if args.kernel_module else None,
        },
        "kernel_source_sha256": hashlib.sha256(
            Path(kernel_module.__file__).read_bytes()
        ).hexdigest(),
        "head_tile": kernel_module._BLOCK_H,
        "autotune_note": "Existing keys omit query count; listed query order retained",
        "cases": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for tokens in args.tokens:
            requests = min(tokens, 5)
            blocks_per_request = args.context // 64
            blocks = requests * blocks_per_request
            table = torch.randperm(blocks, device="cuda").int().reshape(requests, -1)
            request_ids = (
                torch.arange(tokens, device="cuda") * requests // tokens
            ).int()
            indices = torch.randint(
                args.context, (tokens, 2176), device="cuda", dtype=torch.int32
            )
            indices[:, 2051:] = -1
            q = torch.randn(tokens, 32, 512, device="cuda", dtype=torch.float16) * 0.25
            logical_cache = (
                torch.randn(blocks, 64, 512, device="cuda", dtype=torch.float16) * 0.25
            )
            expected = torch.empty_like(q)
            for start in range(0, tokens, 16):
                end = min(start + 16, tokens)
                idx = indices[start:end].long()
                valid = idx >= 0
                idx = idx.clamp_min(0)
                physical = table[request_ids[start:end, None], idx // 64].long()
                selected = logical_cache[physical, idx % 64].float()
                scores = torch.einsum("thd,tkd->thk", q[start:end].float(), selected)
                scores *= 256**-0.5
                scores.masked_fill_(~valid[:, None, :], -torch.inf)
                expected[start:end] = torch.einsum(
                    "thk,tkd->thd", scores.softmax(-1), selected
                ).half()
            del selected, scores, physical, idx
            for interleave in args.interleaves:
                storage = torch.full(
                    (blocks, 64 * interleave, 512),
                    10000.0,
                    device="cuda",
                    dtype=torch.float16,
                )
                cache = storage[:, :64]
                cache.copy_(logical_cache)
                rows, stride = flat_kv_row_view(cache, 64)

                def run(
                    split,
                    request_ids=request_ids,
                    table=table,
                    indices=indices,
                    stride=stride,
                    q=q,
                    rows=rows,
                    tokens=tokens,
                ):
                    mapped = triton_convert_req_index_to_global_index(
                        request_ids,
                        table,
                        indices,
                        BLOCK_SIZE=64,
                        BLOCK_STRIDE_ROWS=stride,
                        NUM_TOPK_TOKENS=2176,
                    )
                    return kernel_module.triton_mla_sparse_attention(
                        q,
                        rows.view(-1, 1, 512),
                        mapped.view(tokens, 1, 2176),
                        256**-0.5,
                        num_kv_splits=split,
                        sm_count=sm_count,
                    )

                qualified = []
                for split in args.splits:
                    call = functools.partial(run, split)
                    start_time = time.perf_counter()
                    actual = call()
                    delta = actual.float() - expected.float()
                    row = {
                        "tokens": tokens,
                        "interleave": interleave,
                        "requested_splits": split,
                        "effective_splits": split
                        or kernel_module._choose_num_kv_splits(
                            tokens,
                            32 // min(32, kernel_module._BLOCK_H),
                            2176,
                            sm_count,
                        ),
                        "max_abs_error": delta.abs().max().item(),
                        "relative_l2": (delta.norm() / expected.float().norm()).item(),
                        "first_call_seconds": time.perf_counter() - start_time,
                    }
                    report["cases"].append(row)
                    save()
                    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_result = call()
                    graph.replay()
                    torch.testing.assert_close(
                        graph_result, expected, atol=3e-4, rtol=3e-3
                    )
                    row["correct"] = row["graph_correct"] = True
                    kernel = (
                        kernel_module._sparse_mla_kernel_final
                        if row["effective_splits"] == 1
                        else kernel_module._sparse_mla_kernel_split
                    )
                    row["selected_config"] = str(kernel.best_config)
                    qualified.append((row, call))
                    del graph_result, graph, actual, delta
                    save()
                for row, call in qualified:
                    row["timings_ms"] = {
                        mode: measure(call, mode, args.iterations, args.repeats)
                        for mode in ("eager", "graph")
                    }
                    save()
                del qualified, call, run, rows, cache, storage
            del logical_cache, expected, q
        report["complete"] = True
        save()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
