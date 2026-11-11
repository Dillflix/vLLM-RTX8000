# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualify SM75 FP16 dense candidates against the actual F.linear fallback.

Single-rank component timings, not TP communication or serving performance.
Synthetic inputs screen candidates; --captures replays real, untimed worker
inputs before promotion. Events exclude the cache flush in cold-cache runs.
Variant -1 is F.linear; native candidates allocate their output equivalently.
--preallocated compares torch.mm(out=...) with the allocation-free native op.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import regex as re
import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.nvidia.ops.rtx8000 import load_native

SHAPES = (
    (48, 2560),
    (512, 2560),
    (640, 2560),
    (2560, 320),
    (2560, 3072),
    (8192, 2560),
    (7296, 2560),
    (10240, 320),
    (320, 10240),
    (336, 10240),
    (124160, 2560),
)
VARIANTS = {
    -1: "F.linear (torch.mm out= when preallocated)",
    0: "FP32 vector accumulation, 128 threads",
    1: "FP32 vector accumulation, 256 threads",
    2: "CUTLASS SM75 FP32 accumulation, BM32 BN64 BK64",
    3: "CUTLASS SM75 FP32 accumulation, BM32 BN128 BK64",
    4: "WMMA FP32 accumulation, BM16 BN64 BK64 skew0",
    5: "WMMA FP32 accumulation, BM16 BN64 BK128 skew0",
    6: "WMMA FP32 accumulation, BM16 BN128 BK64 skew0",
    7: "WMMA FP32 accumulation, BM16 BN128 BK128 skew0",
    8: "WMMA FP32 accumulation, BM16 BN64 BK128 skew8",
    9: "WMMA FP32 accumulation, BM16 BN64 BK128 skew16",
    10: "WMMA FP32 accumulation, BM16 BN128 BK64 skew8",
    11: "WMMA FP32 accumulation, BM16 BN128 BK64 skew16",
}


def operating_snapshot():
    """Instantaneous operating state, not sampled clocks during each kernel."""
    return subprocess.check_output(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,uuid,pstate,temperature.gpu,power.draw,power.limit,"
                "clocks.sm,clocks.mem"
            ),
            "--format=csv",
        ],
        text=True,
    ).strip()


class CaptureWorkerExtension:
    """Diagnostic-only hook; never use timings from this worker.

    Set RTX8000_DENSE_CAPTURE_DIR and RTX8000_DENSE_CAPTURE_RANK (default 2).
    After health, arm with arm.json, e.g. {"label":"mtp5","tokens":[4,8,12,16]}.
    An optional "shapes" list contains [N,K] pairs. One real parameter per shape is
    selected; its weights are saved once and input rows once per label/M. Eager
    serving is required: graph replay does not re-enter Python. Removing the marker
    disarms capture. Existing native dispatch is not changed by this hook.
    """

    def init_worker_extension(self):
        directory = os.environ.get("RTX8000_DENSE_CAPTURE_DIR")
        rank = int(os.environ.get("RTX8000_DENSE_CAPTURE_RANK", "2"))
        if not directory or self.rank != rank:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "arm.json"
        original = F.linear
        selected, saved, observed = {}, set(), {}

        def wrapped(input, weight, bias=None):
            if (
                not marker.exists()
                or torch.cuda.is_current_stream_capturing()
                or bias is not None
                or input.ndim != 2
                or weight.ndim != 2
                or input.dtype != torch.float16
                or weight.dtype != torch.float16
                or not input.is_contiguous()
                or not weight.is_contiguous()
                or tuple(weight.shape) not in SHAPES
            ):
                return original(input, weight, bias)
            request = json.loads(marker.read_text())
            label = request["label"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
                raise ValueError("Dense capture label must be a safe file component")
            n, k = weight.shape
            shape = (n, k)
            if "shapes" in request and [n, k] not in request["shapes"]:
                return original(input, weight, bias)
            if shape not in selected:
                named = self.model_runner.model.named_parameters()
                names = [name for name, p in named if p.data_ptr() == weight.data_ptr()]
                speculator = getattr(self.model_runner, "speculator", None)
                draft_model = getattr(speculator, "model", None)
                if draft_model is not None:
                    names += [
                        "draft." + name
                        for name, p in draft_model.named_parameters()
                        if p.data_ptr() == weight.data_ptr()
                    ]
                if not names:
                    return original(input, weight, bias)
                selected[shape] = (weight.data_ptr(), names)
            pointer, names = selected[shape]
            if pointer != weight.data_ptr():
                return original(input, weight, bias)
            m = input.shape[0]
            observed.setdefault(label, {}).setdefault(f"{n}x{k}", set()).add(m)
            (directory / f"observed-{label}-rank{rank}.json").write_text(
                json.dumps(
                    {
                        "request": request,
                        "shapes": {
                            name: sorted(values)
                            for name, values in observed[label].items()
                        },
                    },
                    indent=2,
                )
            )
            key = (label, n, k, m)
            if m not in request["tokens"] or key in saved:
                return original(input, weight, bias)
            metadata = {
                **request,
                "rank": rank,
                "tp": self.parallel_config.tensor_parallel_size,
                "pp": self.parallel_config.pipeline_parallel_size,
                "model": str(self.model_config.model),
                "parameter_names": names,
                "actual_tokens": m,
                "shape": [n, k],
                "dtype": str(weight.dtype),
            }
            weights_path = directory / f"weights-{n}x{k}-rank{rank}.pt"
            if shape not in saved:
                if weights_path.exists():
                    raise FileExistsError(f"Refusing to overwrite {weights_path}")
                torch.save(
                    {"weight": weight.detach().cpu(), "metadata": metadata},
                    weights_path,
                )
                saved.add(shape)
            destination = directory / f"{label}-{n}x{k}-m{m}-rank{rank}.pt"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            torch.save(
                {
                    "input": input.detach().cpu(),
                    "weights_file": weights_path.name,
                    "metadata": metadata,
                },
                destination,
            )
            saved.add(key)
            return original(input, weight, bias)

        F.linear = wrapped


def time_calls(fn, flush, cold, mode, samples, repeats, graph_warmups=0):
    for _ in range(5):
        fn()
    torch.accelerator.synchronize()
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=mode == "graph"),
            torch.cuda.Event(enable_timing=True, external=mode == "graph"),
        )
        for _ in range(samples)
    ]

    def sequence():
        for start, end in events:
            if cold:
                flush.zero_()
            start.record()
            fn()
            end.record()

    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sequence()
        for _ in range(graph_warmups):
            graph.replay()
        if graph_warmups:
            torch.accelerator.synchronize()
    values = []
    for _ in range(repeats):
        if graph is None:
            sequence()
        else:
            graph.replay()
        torch.accelerator.synchronize()
        values.extend(start.elapsed_time(end) for start, end in events)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def time_graph(fn, flush, cold):
    """Retain the shared helper API used by the QSA component benchmark."""
    values = time_calls(fn, flush, cold, "graph", 10, 5)["samples_ms"]
    return statistics.median(
        statistics.median(values[start : start + 10]) for start in range(0, 50, 10)
    )


def cases(args):
    if args.captures:
        for path in args.captures:
            item = torch.load(path, map_location="cpu", weights_only=True)
            weight_path = path.parent / item["weights_file"]
            weights = torch.load(weight_path, map_location="cpu", weights_only=True)
            for key in ("rank", "tp", "pp", "model", "parameter_names", "shape"):
                if item["metadata"][key] != weights["metadata"][key]:
                    raise ValueError(
                        f"Capture/weight provenance mismatch: {path}: {key}"
                    )
            yield (
                item["input"].cuda(),
                weights["weight"].cuda(),
                {
                    "capture": str(path),
                    "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "weights": str(weight_path),
                    "weights_sha256": hashlib.sha256(
                        weight_path.read_bytes()
                    ).hexdigest(),
                    **item["metadata"],
                },
            )
    else:
        for n, k in args.shapes:
            weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
            for m in args.tokens:
                x = torch.randn(m, k, device="cuda", dtype=torch.float16)
                yield x, weight, {"synthetic": True, "seed": 1234}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 8, 12, 16])
    parser.add_argument("--shapes", nargs="+", default=[f"{n}x{k}" for n, k in SHAPES])
    parser.add_argument("--variants", nargs="+", type=int, default=list(range(-1, 12)))
    parser.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument(
        "--caches", nargs="+", choices=["hot", "cold"], default=["hot", "cold"]
    )
    parser.add_argument("--orders", type=int, choices=[1, 2], default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph-warmups", type=int, default=3)
    parser.add_argument("--preallocated", action="store_true")
    parser.add_argument("--captures", nargs="+", type=Path)
    args = parser.parse_args()
    args.shapes = [tuple(map(int, value.split("x"))) for value in args.shapes]
    if any(len(shape) != 2 for shape in args.shapes):
        parser.error("Shapes must be NxK")
    torch.accelerator.set_device_index(args.device)
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    assert torch.cuda.get_device_capability() == (7, 5)
    torch.backends.cuda.matmul.allow_tf32 = False
    load_native()
    binary = Path(sys.modules["vllm._rtx8000_C"].__file__)
    flush = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    result = {
        "scope": "one TP shard dense component; no communication or serving",
        "variant_definitions": VARIANTS,
        "gpu": torch.cuda.get_device_name(),
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "native_binary": str(binary),
        "native_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "operating_snapshot_before": operating_snapshot(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "captures"
        },
        "rows": [],
        "failures": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for x, weight, metadata in cases(args):
        m, k = x.shape
        n = weight.shape[0]
        reference = (x.float() @ weight.float().T).half()
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        for order in range(args.orders):
            variants = args.variants if order == 0 else list(reversed(args.variants))
            for variant in variants:
                row = {
                    "M": m,
                    "N": n,
                    "K": k,
                    "variant": variant,
                    "order": order,
                    "metadata": metadata,
                }
                if variant >= 4 and m > 16:
                    continue

                def run(variant=variant, x=x, weight=weight, out=out):
                    if variant == -1:
                        return (
                            torch.mm(x, weight.T, out=out)
                            if args.preallocated
                            else F.linear(x, weight)
                        )
                    y = out if args.preallocated else torch.empty_like(out)
                    torch.ops._rtx8000_C.dense(x, weight, y, variant)
                    return y

                try:
                    out.fill_(float("nan"))
                    if variant >= 0:
                        torch.ops._rtx8000_C.dense(x, weight, out, variant)
                        actual = out
                    else:
                        actual = run()
                    torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-3)
                    error = actual.float() - reference.float()
                    relative_rms = (
                        error.square().mean().sqrt()
                        / reference.float().square().mean().sqrt().clamp_min(1e-12)
                    ).item()
                    assert relative_rms <= 1e-3, relative_rms
                    row.update(
                        max_abs_error=error.abs().max().item(),
                        relative_rms_error=relative_rms,
                    )
                    if n == 124160:
                        row["head_local_argmax_agreement"] = (
                            (actual.argmax(-1) == reference.argmax(-1))
                            .float()
                            .mean()
                            .item()
                        )
                    for mode in args.modes:
                        for cache in args.caches:
                            row[f"{mode}_{cache}"] = time_calls(
                                run,
                                flush,
                                cache == "cold",
                                mode,
                                args.samples,
                                args.repeats,
                                args.graph_warmups,
                            )
                    result["rows"].append(row)
                except Exception as error:
                    row["error"] = str(error)
                    result["failures"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(result, indent=2))
    result["complete"] = True
    result["operating_snapshot_after"] = operating_snapshot()
    args.output.write_text(json.dumps(result, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
