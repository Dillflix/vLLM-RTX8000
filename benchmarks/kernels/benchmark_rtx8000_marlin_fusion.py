# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify paired gate/up+SiLU using saved real-model weights and routes.

Only M16/M32 K64/N128 are instantiated. A fresh unfused control distinguishes
fusion effects from compiler/source changes. Production binaries stay intact.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from benchmark_rtx8000_marlin import (
    PROVENANCE_KEYS,
    Config,
    Replay,
    measure,
    verify_weights,
)

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _load_rtx8000_marlin,
)


def errors(actual, reference):
    difference = actual.float() - reference.float()
    return {
        "max_abs_error": difference.abs().max().item(),
        "relative_l2": (
            difference.norm() / reference.float().norm().clamp_min(1e-12)
        ).item(),
        "reference_rms": reference.float().square().mean().sqrt().item(),
        "reference_max_abs": reference.abs().max().item(),
        "bit_exact": torch.equal(actual, reference),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--dispatcher-library", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verify-weights-from", type=Path, nargs="+", default=[])
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--orders", type=int, default=2, choices=[1, 2])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    os.environ["VLLM_MARLIN_MOE_EXACT_SHARED_MEM"] = "0"
    os.environ["VLLM_RTX8000_MARLIN_TILES"] = "0"
    torch.set_num_threads(2)
    torch.accelerator.set_device_index(args.device)
    assert torch.cuda.get_device_capability() == (7, 5)
    _load_rtx8000_marlin()
    torch.ops.load_library(str(args.library.resolve()))
    original_gemm = torch.ops._moe_C.rtx8000_moe_wna16_marlin_gemm
    legacy_gemm = torch.ops._moe_C.moe_wna16_marlin_gemm
    tuning_gemm = None
    if args.dispatcher_library:
        torch.ops.load_library(str(args.dispatcher_library.resolve()))
        tuning_gemm = torch.ops._rtx8000_marlin_tuning.rtx8000_marlin_tuning_gemm
    prototype = torch.ops._rtx8000_marlin_fusion
    saved_weights = torch.load(args.weights, map_location="cpu", weights_only=False)
    verified = [str(args.weights.resolve())] + verify_weights(
        saved_weights, args.verify_weights_from
    )
    weights = {
        key: saved_weights[key].cuda() for key in ("w1", "w2", "w1_scale", "w2_scale")
    }
    result = {
        "scope": "one captured layer / one TP shard; no serving speedup claim",
        "precision_changed": False,
        "weights_requantized": False,
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weights": str(args.weights),
        "verified_weight_files": verified,
        "dispatcher_library": str(args.dispatcher_library),
        "dispatcher_sha256": (
            hashlib.sha256(args.dispatcher_library.read_bytes()).hexdigest()
            if args.dispatcher_library
            else None
        ),
        "parameters": {
            "orders": args.orders,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "smoke_only": args.smoke_only,
        },
        "operating_snapshot_before": subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=index,uuid,temperature.gpu,power.draw,power.limit,"
                    "clocks.sm,clocks.mem"
                ),
                "--format=csv",
            ],
            text=True,
        ).strip(),
        "library": str(args.library),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resources": {},
        "rows": [],
        "complete": False,
    }
    manifest = args.library.parent / "source-manifest.json"
    if manifest.exists():
        result["source_manifest"] = json.loads(manifest.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2))

    for block_m in (16, 32):
        result["resources"][block_m] = {}
        for fused in (False, True):
            values = prototype.resources(block_m, args.device, fused)
            result["resources"][block_m]["fused" if fused else "control"] = dict(
                zip(("registers", "local_bytes", "shared_bytes", "active_ctas"), values)
            )
    save()
    for path in args.captures:
        capture = torch.load(path, map_location="cpu", weights_only=False)
        capture_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert capture["activation"] == "silu"
        activation = capture["activation_config"]
        assert activation["clamp_limit"] is None
        assert activation["alpha"] == 1 and activation["beta"] == 0
        assert capture["metadata"]["activation_callback"] == (
            "vllm.model_executor.layers.fused_moe.modular_kernel."
            "FusedMoEExpertsModular.activation"
        )
        for key in PROVENANCE_KEYS:
            assert capture["metadata"][key] == saved_weights["metadata"][key]
        if path.parent.resolve() != args.weights.parent.resolve():
            capture_weights = path.parent / (
                f"weights-rank{saved_weights['metadata']['rank']}.pt"
            )
            assert str(capture_weights.resolve()) in verified, (
                f"Add --verify-weights-from {capture_weights} before combining captures"
            )
        replay = Replay(weights, capture, original_gemm)
        assert replay.m in (1, 1600, 2048)
        candidate_config = Config(16 if replay.m == 1 else 32, exact=True)
        production_config = Config(48) if replay.m == 1600 else candidate_config
        if replay.m == 1600:
            assert tuning_gemm is not None, "M1600 needs the isolated dispatcher"

        def select(
            variant,
            replay=replay,
            production_config=production_config,
            candidate_config=candidate_config,
        ):
            legacy = replay.m == 1600 and variant == "production"
            replay.config = production_config if legacy else candidate_config
            replay.gemm = (
                (legacy_gemm if legacy else tuning_gemm)
                if replay.m == 1600
                else original_gemm
            )
            os.environ["VLLM_MARLIN_MOE_EXACT_SHARED_MEM"] = (
                "1" if replay.m == 1600 and not legacy else "0"
            )

        def gate(variant, replay=replay):
            if variant in ("production", "tiled"):
                replay.matrix(False)
                replay.activate()
            else:
                fused = variant == "fused"
                prototype.gate_up(
                    replay.x,
                    replay.mid if fused else replay.gate,
                    weights["w1"],
                    weights["w1_scale"],
                    replay.workspace,
                    *replay.routes,
                    fused,
                )
                if not fused:
                    replay.activate()

        def layer(variant, replay=replay, gate=gate):
            replay.align()
            gate(variant)
            replay.matrix(True)
            ops.moe_sum(
                replay.down.view(replay.m, replay.topk, replay.k), replay.output
            )
            return replay.output

        # Compare the unchanged recompiled kernel to the same M32/exact tile.
        # M1600 separately retains the actual production M48/legacy baseline.
        select("tiled")
        replay.align()
        candidate_routes = replay.routes
        gate("tiled")
        candidate_activation = replay.mid.clone()
        replay.matrix(True)
        ops.moe_sum(replay.down.view(replay.m, replay.topk, replay.k), replay.output)
        candidate_output = replay.output.clone()
        select("production")
        if replay.m == 1600:
            replay.align()
        production_routes = replay.routes
        gate("production")
        reference_activation = replay.mid.clone()
        replay.matrix(True)
        ops.moe_sum(replay.down.view(replay.m, replay.topk, replay.k), replay.output)
        reference_output = replay.output.clone()
        for order in range(args.orders):
            variants = ["production", "control", "fused"]
            if replay.m == 1600:
                variants.insert(1, "tiled")
            if order:
                variants.reverse()
            for variant in variants:
                select(variant)
                row = {
                    "capture": str(path),
                    "capture_sha256": capture_sha256,
                    "metadata": capture["metadata"],
                    "tokens": replay.m,
                    "block_m": replay.config.block_m,
                    "exact_shared_memory": replay.config.exact,
                    "variant": variant,
                    "order": order,
                }
                try:
                    replay.routes = (
                        production_routes
                        if variant == "production"
                        else candidate_routes
                    )
                    assert torch.count_nonzero(replay.workspace).item() == 0
                    replay.gate.fill_(float("nan"))
                    replay.mid.fill_(float("nan"))
                    gate(variant)
                    assert torch.isfinite(replay.mid).all().item()
                    assert torch.count_nonzero(replay.workspace).item() == 0
                    row["activation_errors"] = errors(replay.mid, reference_activation)
                    torch.testing.assert_close(
                        replay.mid, reference_activation, atol=0.04, rtol=0
                    )
                    if variant == "control":
                        row["same_tile_activation_errors"] = errors(
                            replay.mid, candidate_activation
                        )
                        assert row["same_tile_activation_errors"]["bit_exact"], (
                            "Recompiled control differs from installed kernels"
                        )
                    replay.down.fill_(float("nan"))
                    replay.output.fill_(float("nan"))
                    replay.matrix(True)
                    ops.moe_sum(
                        replay.down.view(replay.m, replay.topk, replay.k), replay.output
                    )
                    assert torch.isfinite(replay.output).all().item()
                    assert torch.count_nonzero(replay.workspace).item() == 0
                    row["output_errors"] = errors(replay.output, reference_output)
                    if variant == "control":
                        row["same_tile_output_errors"] = errors(
                            replay.output, candidate_output
                        )
                        assert row["same_tile_output_errors"]["bit_exact"], (
                            "Recompiled control changes the same-tile layer output"
                        )
                    torch.testing.assert_close(
                        replay.output, reference_output, atol=0.04, rtol=0
                    )
                    row["timings"] = {}
                    if not args.smoke_only:
                        for stage in ("gate_activation", "layer"):
                            run = (
                                (lambda variant=variant: gate(variant))
                                if stage == "gate_activation"
                                else (lambda variant=variant: layer(variant))
                            )
                            row["timings"][stage] = {
                                mode: measure(run, mode, args.iterations, args.repeats)
                                for mode in ("eager", "graph")
                            }
                    row["status"] = "passed"
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                    result["rows"].append(row)
                    save()
                    raise
                result["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
