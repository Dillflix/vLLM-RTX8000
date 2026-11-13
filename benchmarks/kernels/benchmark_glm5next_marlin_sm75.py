# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify and time a synthetic GLM-5.3-Flash TP2 Marlin expert shard.

This uses random weights, not checkpoint weights. Results are milliseconds per
expert layer on one GPU, not serving throughput, user latency, or TP2/PP2 speed.
The 288-expert W4A16 group-128 fixture is quantized once and reused across token
counts. All cases must match dequantized PyTorch references before any timing.

The current source selects local FP16 Tensor Core accumulation and FP32 split-K
reduction. The latter does not imply FP32 MMA accumulation; the loaded binary's
hash is recorded separately for instruction-audit provenance. Optional --library
loads the isolated _glm53_marlin_fp32::gemm candidate only in this process;
installed libraries are unchanged. Run from the source checkout with its test
dependencies installed; reuse of the existing numerical fixture is intentional.
"""

import argparse
import contextlib
import functools
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from benchmark_rtx8000_marlin import measure

import vllm
from tests.kernels.moe.test_moe import MarlinMoEWeightData
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.scalar_type import scalar_types

HIDDEN, INTERMEDIATE, EXPERTS, TOPK, GROUP_SIZE = 4096, 1024, 288, 8, 128


@contextlib.contextmanager
def select_marlin_library(path):
    if path is None:
        yield
        return
    torch.ops.load_library(str(path))
    original = torch.ops._moe_C.moe_wna16_marlin_gemm
    # Keep the existing Python wrapper's defaults and ScalarType conversion.
    # Only this benchmark process's Python operator lookup is replaced.
    torch.ops._moe_C.moe_wna16_marlin_gemm = torch.ops._glm53_marlin_fp32.gemm
    try:
        yield
    finally:
        torch.ops._moe_C.moe_wna16_marlin_gemm = original


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 5, 512, 1024, 2048]
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--library", type=Path, help="Isolated FP32 MMA candidate .so")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.tokens, args.iterations, args.repeats) <= 0:
        parser.error("Token counts, iterations and repeats must be positive")
    torch.accelerator.set_device_index(args.device)
    if torch.cuda.get_device_capability() != (7, 5):
        parser.error("This benchmark qualifies the SM75 path")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extension = args.library or (
        Path(vllm.__file__).parent / "_moe_C_stable_libtorch.abi3.so"
    )
    report = {
        "scope": "synthetic single-layer TP2 shard; no checkpoint or serving claim",
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "native_extension": {
            "path": str(extension),
            "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        },
        "shape": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "topk": TOPK,
        },
        "quantization": {"type": "symmetric INT4 W4A16", "group_size": GROUP_SIZE},
        "precision": {
            "local_mma": "FP32 candidate" if args.library else "FP16 baseline",
            "split_k_reduction": "FP32",
            "basis": "source policy; audit recorded binary separately",
        },
        "routing": "synthetic sigmoid top8, normalized, scale2.5; selection not timed",
        "tolerance": {
            "atol": 0.04,
            "rtol": 0,
            "source": "tests/kernels/moe/test_moe.py::test_fused_marlin_moe",
        },
        "parameters": {
            "tokens": args.tokens,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "cases": [],
        "complete": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with (
            set_current_vllm_config(VllmConfig()),
            select_marlin_library(args.library),
        ):
            start = time.perf_counter()
            w1 = MarlinMoEWeightData.make(
                torch.randn(
                    EXPERTS,
                    2 * INTERMEDIATE,
                    HIDDEN,
                    dtype=torch.float16,
                    device="cuda",
                )
                / 10,
                scalar_types.uint4b8,
                GROUP_SIZE,
            )
            w2 = MarlinMoEWeightData.make(
                torch.randn(
                    EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.float16, device="cuda"
                )
                / 10,
                scalar_types.uint4b8,
                GROUP_SIZE,
            )
            reference_weights = (w1.w_ref, w2.w_ref)
            kwargs = dict(
                w1=w1.qweight,
                w2=w2.qweight,
                bias1=None,
                bias2=None,
                w1_scale=w1.scales,
                w2_scale=w2.scales,
                quant_type_id=scalar_types.uint4b8.id,
                global_num_experts=EXPERTS,
                activation_config=ApplyMoEActivationConfig(),
            )
            del w1, w2
            torch.accelerator.synchronize()
            report["fixture_seconds"] = time.perf_counter() - start
            report["packed_shapes"] = {
                name: list(kwargs[name].shape)
                for name in ("w1", "w2", "w1_scale", "w2_scale")
            }
            runs = []
            for tokens in args.tokens:
                x = torch.randn(tokens, HIDDEN, dtype=torch.float16, device="cuda") / 10
                scores = torch.randn(tokens, EXPERTS, device="cuda").sigmoid()
                weights, ids = scores.topk(TOPK, dim=-1)
                weights = (weights / weights.sum(-1, keepdim=True) * 2.5).contiguous()
                ids = ids.to(torch.int32).contiguous()
                run = functools.partial(
                    fused_marlin_moe,
                    hidden_states=x,
                    topk_weights=weights,
                    topk_ids=ids,
                    **kwargs,
                )
                expected = torch_experts(
                    x,
                    *reference_weights,
                    topk_weight=weights,
                    topk_ids=ids,
                    global_num_experts=EXPERTS,
                )
                actual = run()
                delta = actual.float() - expected.float()
                row = {
                    "tokens": tokens,
                    "max_abs_error": delta.abs().max().item(),
                    "relative_l2": (
                        delta.norm() / expected.float().norm().clamp_min(1e-12)
                    ).item(),
                    "correct": False,
                }
                report["cases"].append(row)
                save()
                torch.testing.assert_close(actual, expected, atol=0.04, rtol=0)
                row["correct"] = True
                if not args.qualify_only:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_actual = run()
                    graph.replay()
                    torch.testing.assert_close(
                        graph_actual, expected, atol=0.04, rtol=0
                    )
                    row["graph_correct"] = True
                    del graph_actual, graph
                runs.append(run)
                del actual, expected, delta
                save()
            # Reference tensors are not part of the timed kernel working set.
            del reference_weights
            torch.accelerator.empty_cache()
            torch.accelerator.reset_peak_memory_stats()
            if not args.qualify_only:
                for row, run in zip(report["cases"], runs):
                    row["timings_ms"] = {
                        mode: measure(run, mode, args.iterations, args.repeats)
                        for mode in ("eager", "graph")
                    }
                    save()
            report["peak_allocated_bytes_after_qualification"] = (
                torch.accelerator.max_memory_allocated()
            )
            report["complete"] = True
            save()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
