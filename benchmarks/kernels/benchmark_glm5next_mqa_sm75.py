# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare indexer query storage formats; excludes FWHT, quantization and top-k."""

import argparse
import functools
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.models.glm5next.nvidia.ops.mqa_logits_sm75 import (
    fp8_mqa_logits_sm75,
    fp8_paged_mqa_logits_sm75,
)
from vllm.triton_utils import triton


def latency(run, repeats):
    warmup_end = time.perf_counter() + 0.2
    while time.perf_counter() < warmup_end:
        run()
        torch.accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--index-kpool", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(0)
    results = []
    for context in (4096, 32768):
        keys = torch.randint(-4, 5, (context, 128)).to(torch.float8_e4m3fn).cuda()
        scales = torch.full((context,), 0.125, device="cuda")
        cache = torch.empty((context // 64, 64 * 132), dtype=torch.uint8, device="cuda")
        cache[:, : 64 * 128] = keys.view(torch.uint8).view(-1, 64 * 128)
        cache[:, 64 * 128 :].view(torch.float32).copy_(scales.view(-1, 64))
        for tokens in (1, 5, 512, 2048):
            q8 = torch.randint(-4, 5, (tokens, args.heads, 128)).to(torch.float8_e4m3fn)
            q16 = q8.to(torch.float16).cuda()
            q8 = q8.cuda()
            weights = torch.rand(tokens, args.heads, device="cuda") / args.heads
            starts = torch.zeros(tokens, dtype=torch.int32, device="cuda")
            ends = torch.full((tokens,), context, dtype=torch.int32, device="cuda")
            table = torch.arange(context // 64, dtype=torch.int32, device="cuda")
            table = table.unsqueeze(0).expand(tokens, -1).contiguous()
            for dtype, queries in (("fp8", q8), ("fp16", q16)):
                if tokens <= 5:
                    run = functools.partial(
                        fp8_paged_mqa_logits_sm75,
                        queries.view(tokens, 1, args.heads, 128),
                        cache.view(-1, 64, 1, 132),
                        weights,
                        ends.view(tokens, 1),
                        table,
                        context,
                    )
                else:
                    run = functools.partial(
                        fp8_mqa_logits_sm75,
                        queries,
                        (keys, scales),
                        weights,
                        starts,
                        ends,
                        context,
                    )

                eager_ms = latency(run, args.repeats)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                graph_ms = latency(graph.replay, args.repeats)
                assert captured.dtype == torch.float32
                row = dict(
                    phase="paged_decode" if tokens <= 5 else "prefill",
                    tokens=tokens,
                    pooled_keys=context,
                    equivalent_context_tokens=context * args.index_kpool,
                    heads=args.heads,
                    query_dtype=dtype,
                    eager_ms=eager_ms,
                    graph_ms=graph_ms,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                del graph, captured
    report = {
        "device": torch.cuda.get_device_name(),
        "triton_version": triton.__version__,
        "triton_module": triton.__file__,
        "scope": "warm MQA logits only; excludes FWHT, quantization, top-k, serving",
        "same_query_values": True,
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
