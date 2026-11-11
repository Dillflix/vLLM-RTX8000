# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay captured SM75 per-channel W8A16 MoE layers without requantization.

This is a single-rank kernel benchmark, not a serving or TP-collective benchmark.
Weights file: CPU tensors w1, w2, w1_scale, w2_scale, plus optional metadata.
Each capture: hidden_states, topk_ids, topk_weights, optional metadata, activation
(default "silu"), activation_config (dict). Capture these from _fused_marlin_moe;
only the no-bias, no-zero-point, no-expert-map, output-router-weight path is used.

--build-dispatcher compiles only the modified dispatcher in an isolated namespace,
reusing the installed library's kernel selector and already compiled kernels. It
does not replace the installed vLLM library. --build-only performs no GPU work.
"""

import argparse
import contextlib
import ctypes
import hashlib
import inspect
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import regex as re
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.scalar_type import scalar_types

EXACT_ENV = "VLLM_MARLIN_MOE_EXACT_SHARED_MEM"
TILES = ((64, 128), (128, 128), (64, 256), (128, 64))  # K, N
WEIGHT_KEYS = ("w1", "w2", "w1_scale", "w2_scale")
PROVENANCE_KEYS = ("layer_parameter", "model", "rank", "tp", "pp")


def capture_activation(values):
    """Accept only the standard activation or its verified modular wrapper."""
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsModular,
    )

    callback = values["activation_func"]
    declared_config = values["activation_config"]
    if callback is None:
        return declared_config or ApplyMoEActivationConfig(), "apply_moe_activation"
    implementation = getattr(callback, "__func__", None)
    if implementation is not FusedMoEExpertsModular.activation:
        raise ValueError("Cannot replay a custom Marlin activation callback")
    effective_config = callback.__self__.activation_config
    if declared_config is not None and declared_config != effective_config:
        raise ValueError("Marlin activation callback/config mismatch")
    identity = f"{implementation.__module__}.{implementation.__qualname__}"
    return effective_config, identity


def verify_weights(saved: dict, paths: list[Path]):
    """Prove separate serving captures use identical packed weights and shard."""
    metadata = saved["metadata"]
    for key in PROVENANCE_KEYS:
        if key not in metadata or metadata[key] in (None, "unknown"):
            raise ValueError(f"Missing weight provenance: {key}")
    verified = []
    for path in paths:
        other = torch.load(path, map_location="cpu", weights_only=True)
        for key in PROVENANCE_KEYS:
            if other.get("metadata", {}).get(key) != metadata[key]:
                raise ValueError(f"Weight provenance differs for {key}: {path}")
        for key in WEIGHT_KEYS:
            a, b = saved[key], other[key]
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                raise ValueError(f"Captured weight tensor differs: {path} / {key}")
        verified.append(str(path.resolve()))
    return verified


class CaptureWorkerExtension:
    """Diagnostic-only worker hook, armed by an explicit JSON marker file.

    Set RTX8000_MARLIN_CAPTURE_DIR and optionally RTX8000_MARLIN_CAPTURE_RANK
    (default 0). After server health, write arm.json in that directory, e.g.
    {"label": "prefill32k", "tokens": [512, 1024, 2048], "context": 32768}.
    Each label/token combination is saved once for the first observed layer.
    Delete the marker to disarm. Use eager serving for decode captures, since
    graph replays do not enter Python. Timings from this server are invalid.
    """

    def init_worker_extension(self):
        import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin

        directory = os.environ.get("RTX8000_MARLIN_CAPTURE_DIR")
        rank = int(os.environ.get("RTX8000_MARLIN_CAPTURE_RANK", "0"))
        if not directory or self.rank != rank:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "arm.json"
        original = marlin._fused_marlin_moe
        signature = inspect.signature(original)
        state = {
            "weight_pointer": None,
            "layer_parameter": None,
            "weights_saved": False,
            "saved": set(),
            "observed": {},
        }

        def wrapped(*args, **kwargs):
            if not marker.exists() or torch.cuda.is_current_stream_capturing():
                return original(*args, **kwargs)
            request = json.loads(marker.read_text())
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            tokens = values["hidden_states"].shape[0]
            label = request["label"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
                raise ValueError("Marlin capture label must be a safe file component")
            key = (label, tokens)
            pointer = values["w1"].data_ptr()
            if state["weight_pointer"] is None:
                # Select during the first armed prefill even for decode-only
                # capture, so an MTP draft layer cannot become the first match.
                state["weight_pointer"] = pointer
                parameters = self.model_runner.model.named_parameters()
                state["layer_parameter"] = next(
                    (
                        name
                        for name, parameter in parameters
                        if parameter.data_ptr() == pointer
                    ),
                    "unknown",
                )
            if state["weight_pointer"] != pointer:
                return original(*args, **kwargs)
            observed = state["observed"].setdefault(label, set())
            if tokens not in observed:
                observed.add(tokens)
                (directory / f"observed-{label}-rank{rank}.json").write_text(
                    json.dumps(
                        {
                            "request": request,
                            "layer_parameter": state["layer_parameter"],
                            "observed_tokens": sorted(observed),
                        },
                        indent=2,
                    )
                )
            if tokens not in request["tokens"] or key in state["saved"]:
                return original(*args, **kwargs)
            for name in (
                "bias1",
                "bias2",
                "w1_zeros",
                "w2_zeros",
                "expert_map",
                "global_scale1",
                "global_scale2",
            ):
                if values[name] is not None:
                    raise ValueError(f"Unsupported captured Marlin feature: {name}")
            if (
                values["apply_router_weight_on_input"]
                or not values["mul_topk_weights_in_gemm2"]
            ):
                raise ValueError("Capture requires router weights in GEMM2")
            if values["quant_type"] != scalar_types.uint8b128:
                raise ValueError("Capture requires per-channel INT8 W8A16")
            if values["topk_ids"] is None:
                raise ValueError("Capture needs unsorted top-k expert IDs")
            activation_config, activation_identity = capture_activation(values)
            metadata = {
                **request,
                "rank": rank,
                "tp": self.parallel_config.tensor_parallel_size,
                "pp": self.parallel_config.pipeline_parallel_size,
                "model": str(self.model_config.model),
                "weight_pointer": pointer,
                "activation_callback": activation_identity,
                "actual_tokens": tokens,
            }
            if not state["weights_saved"]:
                metadata["layer_parameter"] = state["layer_parameter"]
                weights = {
                    name: values[name].detach().cpu().contiguous()
                    for name in WEIGHT_KEYS
                }
                for name in ("w1_scale", "w2_scale"):
                    if weights[name].shape[1] != 1:
                        raise ValueError("Capture requires per-channel scales")
                weights["metadata"] = metadata
                destination = directory / f"weights-rank{rank}.pt"
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite {destination}")
                torch.save(weights, destination)
                state["weights_saved"] = True
            metadata["layer_parameter"] = state["layer_parameter"]
            capture = {
                name: values[name].detach().cpu().contiguous()
                for name in ("hidden_states", "topk_ids", "topk_weights")
            }
            capture["metadata"] = metadata
            capture["activation"] = values["activation"].value
            capture["activation_config"] = asdict(activation_config)
            destination = directory / f"{label}-m{tokens}-rank{rank}.pt"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            torch.save(capture, destination)
            state["saved"].add(key)
            return original(*args, **kwargs)

        marlin._fused_marlin_moe = wrapped


def build_dispatcher(directory: Path) -> str:
    """Build a separately named dispatcher; link to the installed kernel selector."""
    from torch.utils.cpp_extension import load

    import vllm

    source_root = Path(__file__).resolve().parents[2]
    source_path = source_root / "csrc/moe/marlin_moe_wna16/ops.cu"
    original = source_path.read_text()
    signature = re.search(
        r"MarlinFuncPtr get_marlin_kernel\(.*?int stages\)", original, re.S
    ).group()
    declaration = (
        "namespace marlin_moe_wna16 {\n"
        "using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);\n" + signature + ";\n}\n"
    )
    source = original.replace(
        "namespace MARLIN_NAMESPACE_NAME {",
        declaration + "\nnamespace MARLIN_NAMESPACE_NAME {",
        1,
    )
    # The old library's sentinel is not exported; ask its selector for an
    # impossible tile to identify it, then map it to this dispatcher's sentinel.
    selector = """
  auto selected = marlin_moe_wna16::get_marlin_kernel(
      a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
      thread_k_blocks, m_block_size_8, has_zp, group_blocks, threads,
      is_zp_float, stages);
  auto unsupported = marlin_moe_wna16::get_marlin_kernel(
      a_type, b_type, c_type, s_type, 0, 0, 0, false, false, -1, 0, false, stages);
  kernel = selected == unsupported ? MarlinDefault : selected;
"""
    assert source.count('#include "kernel_selector.h"') == 1
    source = source.replace('#include "kernel_selector.h"', selector)
    source = source.replace("moe_wna16_marlin_gemm", "rtx8000_marlin_tuning_gemm")
    source = source.replace(
        "STABLE_TORCH_LIBRARY_IMPL(_moe_C,",
        "STABLE_TORCH_LIBRARY_IMPL(_rtx8000_marlin_tuning,",
    )
    bindings = (source_root / "csrc/moe/torch_bindings.cpp").read_text()
    schema_start = bindings.index('"moe_wna16_marlin_gemm(')
    schema_start = bindings.rfind("m.def(", 0, schema_start)
    schema_end = bindings.index(");", schema_start) + 2
    schema_definition = bindings[schema_start:schema_end].replace(
        "moe_wna16_marlin_gemm", "rtx8000_marlin_tuning_gemm", 1
    )
    schema_registration = (
        "STABLE_TORCH_LIBRARY(_rtx8000_marlin_tuning, m) {\n"
        f"  {schema_definition}\n}}\n\n"
    )
    source = source.replace(
        "STABLE_TORCH_LIBRARY_IMPL(_rtx8000_marlin_tuning,",
        schema_registration + "STABLE_TORCH_LIBRARY_IMPL(_rtx8000_marlin_tuning,",
    )
    # Every modified host symbol is separate from the production dispatcher.
    source = "#define MARLIN_NAMESPACE_NAME rtx8000_marlin_tuning\n" + source
    directory.mkdir(parents=True, exist_ok=True)
    generated = directory / "dispatch.cu"
    if not generated.exists() or generated.read_text() != source:
        generated.write_text(source)
    libraries = list(Path(vllm.__file__).parent.glob("_moe_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one installed MoE library: {libraries}")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    os.environ.setdefault("MAX_JOBS", "2")
    library = load(
        name="rtx8000_marlin_tuning",
        sources=[str(generated)],
        extra_include_paths=[
            str(source_root / "csrc"),
            str(source_path.parent),
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        ],
        extra_ldflags=[str(libraries[0])],
        build_directory=str(directory),
        is_python_module=False,
        verbose=True,
    )
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "dispatcher_source": str(source_path),
                "dispatcher_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "kernel_library": str(libraries[0]),
                "kernel_library_sha256": hashlib.sha256(
                    libraries[0].read_bytes()
                ).hexdigest(),
                "isolated_library": str(library),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            indent=2,
        )
    )
    return str(library)


@contextlib.contextmanager
def shared_mode(exact: bool):
    previous = os.environ.get(EXACT_ENV)
    os.environ[EXACT_ENV] = "1" if exact else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(EXACT_ENV, None)
        else:
            os.environ[EXACT_ENV] = previous


@dataclass(frozen=True)
class Config:
    block_m: int
    thread_k: int = -1
    thread_n: int = -1
    blocks_per_sm: int = -1
    exact: bool = False


def scratch_bound(block_m: int, thread_k: int, thread_n: int) -> int:
    """Audited host allocation bound for FP16/per-channel U8B128, two stages.

    M8 deliberately keeps the dispatcher's M16 accounting; this is a safe
    allocation bound, not a claim that all allocated bytes are accessed.
    """
    m = ((block_m + 15) // 16) * 16
    metadata = m * 16
    a = 2 * m * thread_k * 2
    b = 2 * thread_k * thread_n
    reduction = m * (thread_n + 8) * 2
    bias = thread_n * 2
    scales = 2 * thread_n * 2
    return metadata + a + max(b, reduction, min(b, reduction) + bias) + scales


def cuda_resource_limits(device: int) -> dict:
    """Read CUDA's shared-memory limits; this performs no device allocation."""
    names = []
    if os.environ.get("CUDA_HOME"):
        names.append(str(Path(os.environ["CUDA_HOME"]) / "lib64/libcudart.so"))
    names += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    for name in names:
        try:
            runtime = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Cannot locate CUDA runtime for resource queries")
    query = runtime.cudaDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    result = {}
    for name, attribute in (
        ("sms", 16),
        ("shared_bytes_per_sm", 81),
        ("optin_shared_bytes_per_block", 97),
        ("reserved_shared_bytes_per_block", 111),
    ):
        value = ctypes.c_int()
        status = query(ctypes.byref(value), attribute, device)
        if status:
            raise RuntimeError(f"CUDA attribute {attribute} failed: {status}")
        result[name] = value.value
    return result


def existing_tile(block_m: int, thread_k: int, thread_n: int) -> bool:
    if thread_k * thread_n // 64 != 256:
        return True
    return (thread_k, thread_n) == ((128, 128) if block_m <= 16 else (64, 256))


class ResourceDiagnostics:
    """Query occupancy through the dispatcher's checked, non-launching path."""

    def __init__(self, device: int):
        self.limits = cuda_resource_limits(device)
        self.queries = {}

    def query(self, replay, block_m: int, thread_k: int, thread_n: int, down: bool):
        key = (block_m, thread_k, thread_n)
        if key in self.queries:
            return self.queries[key]
        bound = scratch_bound(*key)
        # Four CTAs provably cannot fit for any existing tile in this scope.
        # Therefore the checked C++ request must reject before kernel launch.
        if bound * 4 <= self.limits["shared_bytes_per_sm"]:
            raise ValueError("The non-launching occupancy probe needs a larger bound")
        previous = replay.config, replay.stage, replay.routes
        replay.config = Config(block_m, thread_k, thread_n, 4, True)
        replay.stage = "layer"
        try:
            replay.align()
            with shared_mode(True):
                try:
                    replay.matrix(down)
                except RuntimeError as exc:
                    match = re.search(
                        r"exceeds CUDA occupancy = (\d+) "
                        r"with dynamic shared memory = (\d+)",
                        str(exc),
                    )
                    if match is None:
                        raise RuntimeError(
                            "Occupancy diagnostics require the patched dispatcher"
                        ) from exc
                else:
                    raise AssertionError(
                        "Over-capacity occupancy probe was not rejected"
                    )
        finally:
            replay.config, replay.stage, replay.routes = previous
        active, allocated = map(int, match.groups())
        assert allocated == bound, (allocated, bound)
        assert 1 <= active <= 3, active
        record = {
            "block_m": block_m,
            "thread_k": thread_k,
            "thread_n": thread_n,
            "safe_scratch_bound_bytes": bound,
            "cuda_max_active_ctas_at_scratch_bound": active,
            "source": "CUDA occupancy API via checked prelaunch rejection",
        }
        self.queries[key] = record
        return record

    def launch(self, replay, cfg: Config, down: bool):
        if (down and replay.stage == "gate_up") or (
            not down and replay.stage == "down"
        ):
            cfg = Config(cfg.block_m, exact=cfg.exact)
        n, k = (replay.k, replay.n) if down else (2 * replay.n, replay.k)
        if cfg.thread_k == -1:
            tiles = (
                ((128, 128), (64, 128), (128, 64))
                if cfg.block_m <= 16
                else ((64, 256), (64, 128), (128, 64))
            )
            best = None
            for tk, tn in tiles:
                if k % tk or n % tn:
                    continue
                query = self.query(replay, cfg.block_m, tk, tn, down)
                bound = query["safe_scratch_bound_bytes"]
                if bound > self.limits["optin_shared_bytes_per_block"] - 512:
                    continue
                if cfg.exact:
                    count = query["cuda_max_active_ctas_at_scratch_bound"]
                else:
                    count = self.limits["optin_shared_bytes_per_block"] // (
                        bound + 1536
                    )
                    # At <=255 registers/thread, the legacy register heuristic
                    # permits >=2 CTAs for128 threads, >=1 for256. The shared
                    # limit never exceeds that floor for these existing tiles.
                    register_floor = (255 * 1024) // (255 * (tk * tn // 64) * 4)
                    if count > register_floor:
                        raise ValueError(
                            "Legacy dispatch requires actual register data"
                        )
                count = max(min(count, 4 if cfg.block_m <= 16 else 2), 1)
                work = (n // tn) * replay.m * replay.topk * 4
                if work < self.limits["sms"] * count:
                    count = max(work // self.limits["sms"], 1)
                if best is None or count > best[2]:
                    best = (tk, tn, count)
            assert best is not None
            tk, tn, count = best
            selection = "source-derived automatic dispatcher policy"
        else:
            tk, tn, count = cfg.thread_k, cfg.thread_n, cfg.blocks_per_sm
            selection = "explicit operator arguments"
        query = self.query(replay, cfg.block_m, tk, tn, down)
        bound = query["safe_scratch_bound_bytes"]
        allocated = bound if cfg.exact else self.limits["optin_shared_bytes_per_block"]
        if not cfg.exact and count > 1:
            allocated = allocated // count - 1024
        # This kernel uses only extern dynamic shared memory. Larger legacy
        # allocation can only reduce the already queried occupancy limit.
        possible = min(
            query["cuda_max_active_ctas_at_scratch_bound"],
            self.limits["shared_bytes_per_sm"]
            // (allocated + self.limits["reserved_shared_bytes_per_block"]),
        )
        return {
            **query,
            "allocated_shared_bytes": allocated,
            "grid_ctas_per_sm": count,
            "scratch_fits_allocation": allocated >= bound,
            "possible_ctas_per_sm": possible if allocated >= bound else None,
            "possible_ctas_source": (
                "CUDA occupancy API"
                if cfg.exact
                else "derived from queried occupancy and larger shared allocation"
            ),
            "selection_source": selection,
        }


class Replay:
    def __init__(self, weights, capture, gemm):
        self.weights = weights
        self.x = capture["hidden_states"].cuda().contiguous()
        self.ids = capture["topk_ids"].cuda().contiguous()
        self.tw = capture["topk_weights"].cuda().contiguous()
        self.m, self.k = self.x.shape
        self.topk = self.ids.shape[1]
        self.e = weights["w1"].shape[0]
        self.n = weights["w2"].shape[1] * 16
        self.activation = MoEActivation.from_str(capture.get("activation", "silu"))
        self.activation_config = ApplyMoEActivationConfig(
            **capture.get("activation_config", {})
        )
        assert self.activation.is_gated
        assert self.x.dtype == torch.float16 and self.tw.dtype == torch.float32
        assert self.ids.shape == self.tw.shape == (self.m, self.topk)
        assert self.ids.min() >= 0 and self.ids.max() < self.e
        assert self.k == 2560 and self.n == 320
        assert self.e == 512 and self.topk == 10
        assert weights["w1_scale"].shape == (self.e, 1, 2 * self.n)
        assert weights["w2_scale"].shape == (self.e, 1, self.k)
        for scale in (weights["w1_scale"], weights["w2_scale"]):
            assert scale.dtype == torch.float16
        for packed in (weights["w1"], weights["w2"]):
            assert packed.dtype == torch.int32 and packed.is_contiguous()
        device = self.x.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(sms * 4, device=device, dtype=torch.int32)
        self.cache13 = torch.empty(
            self.m * self.topk * max(self.k, 2 * self.n),
            device=device,
            dtype=self.x.dtype,
        )
        self.gate = self.cache13[: self.m * self.topk * 2 * self.n].view(-1, 2 * self.n)
        self.down = self.cache13[: self.m * self.topk * self.k].view(-1, self.k)
        self.mid = torch.empty(
            self.m * self.topk, self.n, device=device, dtype=self.x.dtype
        )
        self.output = torch.empty_like(self.x)
        self.gemm = gemm
        self.config = Config(8)
        self.stage = "layer"
        self.routes = None

    def align(self):
        self.routes = moe_align_block_size(
            self.ids, self.config.block_m, self.e, ignore_invalid_experts=True
        )

    def matrix(self, down: bool):
        cfg = self.config
        if (down and self.stage == "gate_up") or (not down and self.stage == "down"):
            cfg = Config(cfg.block_m, exact=cfg.exact)
        self.gemm(
            self.mid if down else self.x,
            self.down if down else self.gate,
            self.weights["w2" if down else "w1"],
            None,
            self.weights["w2_scale" if down else "w1_scale"],
            None,
            None,
            self.workspace,
            *self.routes,
            self.tw,
            cfg.block_m,
            1 if down else self.topk,
            down,
            scalar_types.uint8b128.id,
            self.m * self.topk if down else self.m,
            self.k if down else 2 * self.n,
            self.n if down else self.k,
            False,
            True,
            False,
            cfg.thread_k,
            cfg.thread_n,
            cfg.blocks_per_sm,
        )

    def activate(self):
        apply_moe_activation(
            self.activation,
            self.mid,
            self.gate,
            activation_config=self.activation_config,
            topk_ids=self.ids,
        )

    def layer(self):
        self.align()
        self.matrix(False)
        self.activate()
        self.matrix(True)
        ops.moe_sum(self.down.view(self.m, self.topk, self.k), self.output)
        return self.output

    def checked_matrix(self, down: bool):
        assert torch.count_nonzero(self.workspace).item() == 0, "Leaked Marlin locks"
        target = self.down if down else self.gate
        target.fill_(float("nan"))
        self.matrix(down)
        assert torch.isfinite(target).all().item(), "Unwritten/nonfinite Marlin output"
        assert torch.count_nonzero(self.workspace).item() == 0, "Leaked Marlin locks"

    def checked_layer(self, realign: bool = True):
        """Poison reused storage only during untimed correctness checks."""
        self.output.fill_(float("nan"))
        if realign:
            self.align()
        self.checked_matrix(False)
        self.mid.fill_(float("nan"))
        self.activate()
        assert torch.isfinite(self.mid).all().item(), "Unwritten/nonfinite activation"
        # Gate and down share cache13: poison down only after activation used gate.
        self.checked_matrix(True)
        ops.moe_sum(self.down.view(self.m, self.topk, self.k), self.output)
        assert torch.isfinite(self.output).all().item(), "Unwritten/nonfinite MoE sum"
        return self.output

    def reference(self):
        def sum_routes(input, output, topk_ids, expert_map):
            ops.moe_sum(input, output)
            return output

        with shared_mode(False):
            return fused_marlin_moe(
                self.x,
                self.weights["w1"],
                self.weights["w2"],
                None,
                None,
                self.weights["w1_scale"],
                self.weights["w2_scale"],
                self.tw,
                self.ids,
                scalar_types.uint8b128.id,
                activation=self.activation,
                activation_config=self.activation_config,
                moe_sum=sum_routes,
            ).clone()


def measure(fn, mode: str, iterations: int, repeats: int):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                fn()
        run = graph.replay
    else:

        def run():
            for _ in range(iterations):
                fn()

    samples = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def candidates(tokens: int, blocks: list[int], exact_modes: list[bool], stage: str):
    heuristic_m = next(
        (m for m in (8, 16, 32, 48, 64) if tokens * 10 / 512 / m < 0.9), 64
    )
    result = [Config(heuristic_m)]
    for exact in exact_modes:
        for m in blocks:
            result.append(Config(m, exact=exact))
            for k, n in TILES:
                # Match generate_kernels.py's existing 256-thread variants.
                if not existing_tile(m, k, n):
                    continue
                # Separate stage sweeps include gate K128 and down N256;
                # a whole-layer shared choice must divide both GEMMs.
                if stage != "down" and (2560 % k or 640 % n):
                    continue
                if stage != "gate_up" and (320 % k or 2560 % n):
                    continue
                for ctas in (1, 2, 3, 4):
                    result.append(Config(m, k, n, ctas, exact))
    return list(dict.fromkeys(result))


def finalists(report: dict, capture: Path, stage: str, modes: list[str], tokens: int):
    """Keep two leading configs per timing mode and the heuristic baseline."""
    selected = candidates(tokens, [], [], stage)
    rows = [
        row
        for row in report["rows"]
        if Path(row["capture"]).resolve() == capture.resolve()
        and row["stage"] == stage
        and row["status"] == "passed"
    ]
    if not rows:
        raise ValueError(f"No successful screening rows for {capture} / {stage}")
    for mode in modes:
        samples = {}
        for row in rows:
            if mode not in row["timings"]:
                continue
            cfg = Config(**row["config"])
            samples.setdefault(cfg, []).extend(row["timings"][mode]["samples_ms"])
        if not samples:
            raise ValueError(f"No {mode} screening timings for {capture} / {stage}")
        selected.extend(
            sorted(samples, key=lambda cfg: statistics.median(samples[cfg]))[:2]
        )
    return list(dict.fromkeys(selected))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--verify-weights-from", type=Path, nargs="+", default=[])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--captures", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 16, 32, 48, 64])
    parser.add_argument(
        "--shared-modes",
        nargs="+",
        choices=["legacy", "exact"],
        default=["legacy", "exact"],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["layer", "gate_up", "down"],
        default=["layer", "gate_up", "down"],
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--orders", type=int, choices=[1, 2], default=2)
    parser.add_argument("--finalists-from", type=Path)
    parser.add_argument("--build-dispatcher", type=Path)
    parser.add_argument("--dispatcher-library", type=Path)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    library = None
    if args.build_dispatcher:
        library = build_dispatcher(args.build_dispatcher)
    elif args.dispatcher_library:
        library = str(args.dispatcher_library)
        torch.ops.load_library(library)
    if args.build_only:
        if not args.build_dispatcher:
            parser.error("--build-only requires --build-dispatcher")
        return
    if args.verify_only and not args.weights:
        parser.error("--verify-only requires --weights")
    if not args.verify_only and not (args.weights and args.captures and args.output):
        parser.error("--weights, --captures and --output are required for replay")
    if not set(args.blocks).issubset({8, 16, 32, 48, 64}):
        parser.error("Only existing M8/16/32/48/64 kernels are supported")
    torch.set_num_threads(2)
    saved = torch.load(args.weights, map_location="cpu", weights_only=True)
    verified = [str(args.weights.resolve())] + verify_weights(
        saved, args.verify_weights_from
    )
    if args.verify_only:
        result = {"verified_weight_files": verified, "bit_exact": True}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    torch.accelerator.set_device_index(args.device)
    assert torch.cuda.get_device_capability() == (7, 5)
    resources = ResourceDiagnostics(args.device)
    gemm = (
        torch.ops._rtx8000_marlin_tuning.rtx8000_marlin_tuning_gemm
        if library
        else torch.ops._moe_C.moe_wna16_marlin_gemm
    )
    weights = {key: saved[key].cuda().contiguous() for key in WEIGHT_KEYS}
    screening = (
        json.loads(args.finalists_from.read_text()) if args.finalists_from else None
    )
    report = {
        "scope": __doc__,
        "device": torch.cuda.get_device_name(),
        "device_index": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dispatcher_library": library,
        "layer_sampling": (
            "One captured layer per weights file; routing varies across layers. "
            "These timings do not establish full-model or TP2/PP2 performance."
        ),
        "finalists_from": str(args.finalists_from) if args.finalists_from else None,
        "weights": str(args.weights),
        "verified_weight_files": verified,
        "smoke_only": args.smoke_only,
        "resource_limits": resources.limits,
        "weight_metadata": saved.get("metadata", {}),
        "tolerance": {"atol": 0.04, "rtol": 0, "source": "test_fused_marlin_moe"},
        "rows": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["resource_queries"] = list(resources.queries.values())
        args.output.write_text(json.dumps(report, indent=2))

    for path in args.captures:
        capture = torch.load(path, map_location="cpu", weights_only=True)
        for key in PROVENANCE_KEYS:
            if capture.get("metadata", {}).get(key) != saved["metadata"][key]:
                raise ValueError(f"Capture provenance differs for {key}: {path}")
        if path.parent.resolve() != args.weights.parent.resolve():
            capture_weights = (
                path.parent / f"weights-rank{saved['metadata']['rank']}.pt"
            )
            if str(capture_weights.resolve()) not in verified:
                raise ValueError(
                    f"Add --verify-weights-from {capture_weights} before combining runs"
                )
        replay = Replay(weights, capture, gemm)
        reference = replay.reference()
        reference_rms = reference.float().square().mean().sqrt().item()
        reference_max = reference.abs().max().item()
        if args.smoke_only:
            experiments = []
            for m in args.blocks:
                query = resources.query(replay, m, 64, 128, False)
                desired = 3 if m <= 16 else 2 if m <= 48 else 1
                ctas = min(desired, query["cuda_max_active_ctas_at_scratch_bound"])
                experiments.append(("layer", Config(m, 64, 128, ctas, True)))
        else:
            experiments = [
                (stage, cfg)
                for stage in args.stages
                for cfg in (
                    finalists(screening, path, stage, args.modes, replay.m)
                    if screening is not None
                    else candidates(
                        replay.m,
                        args.blocks,
                        [mode == "exact" for mode in args.shared_modes],
                        stage,
                    )
                )
            ]
        for order in range(1 if args.smoke_only else args.orders):
            for stage, cfg in (
                experiments if order == 0 else list(reversed(experiments))
            ):
                replay.config = cfg
                replay.stage = stage
                row = {
                    "capture": str(path),
                    "metadata": capture.get("metadata", {}),
                    "tokens": replay.m,
                    "config": asdict(cfg),
                    "stage": stage,
                    "order": order,
                    "reference_rms": reference_rms,
                    "reference_max_abs": reference_max,
                    "launch_resources": {
                        "gate_up": resources.launch(replay, cfg, False),
                        "down": resources.launch(replay, cfg, True),
                    },
                }
                try:
                    with shared_mode(cfg.exact):
                        actual = replay.checked_layer().clone()
                        torch.accelerator.synchronize()
                        error = actual.float() - reference.float()
                        row["max_abs_error"] = error.abs().max().item()
                        row["relative_l2"] = (
                            error.norm() / reference.float().norm().clamp_min(1e-12)
                        ).item()
                        row["max_error_over_reference_rms"] = row[
                            "max_abs_error"
                        ] / max(reference_rms, 1e-12)
                        torch.testing.assert_close(actual, reference, atol=0.04, rtol=0)
                        if cfg.exact and cfg.thread_k != -1:
                            comparison_routes = moe_align_block_size(
                                replay.ids,
                                cfg.block_m,
                                replay.e,
                                ignore_invalid_experts=True,
                            )
                            padded = replay.routes[2].item()
                            row["realignment_bit_exact"] = torch.equal(
                                replay.routes[0][:padded],
                                comparison_routes[0][:padded],
                            )
                            row["same_grid_routes_frozen"] = True
                            if stage != "layer":
                                replay.checked_matrix(stage == "down")
                                shared_only = (
                                    replay.down if stage == "down" else replay.gate
                                ).clone()
                            else:
                                shared_only = actual
                            try:
                                with shared_mode(False):
                                    if stage == "layer":
                                        same_grid_legacy = replay.checked_layer(
                                            realign=False
                                        ).clone()
                                    else:
                                        replay.checked_matrix(stage == "down")
                                        same_grid_legacy = (
                                            replay.down
                                            if stage == "down"
                                            else replay.gate
                                        ).clone()
                            except RuntimeError as exc:
                                if "Invalid thread config" not in str(exc):
                                    raise
                                row["same_grid_legacy"] = (
                                    "cannot fit legacy shared-memory budget"
                                )
                            else:
                                row["same_grid_bit_exact"] = torch.equal(
                                    shared_only, same_grid_legacy
                                )
                                assert row["same_grid_bit_exact"], (
                                    "Shared allocation alone must preserve bits"
                                )
                        assert torch.count_nonzero(replay.workspace).item() == 0
                        row["padded_routes"] = replay.routes[2].item()
                        row["timings"] = {}
                        if stage == "layer":
                            run = replay.layer
                        else:

                            def run(stage=stage, replay=replay):
                                replay.matrix(stage == "down")

                        if not args.smoke_only:
                            for mode in args.modes:
                                row["timings"][mode] = measure(
                                    run, mode, args.iterations, args.repeats
                                )
                        row["status"] = "passed"
                except RuntimeError as exc:
                    message = str(exc)
                    if not any(
                        part in message
                        for part in (
                            "Invalid thread config",
                            "Unsupported shapes",
                            "exceeds CUDA occupancy",
                        )
                    ):
                        save()
                        raise
                    row.update(status="unsupported", reason=message)
                except AssertionError as exc:
                    row.update(status="numerical_failure", reason=str(exc))
                report["rows"].append(row)
                if (
                    len(report["rows"]) == 1
                    or len(report["rows"]) % 32 == 0
                    or row["status"] == "numerical_failure"
                ):
                    save()
                print(json.dumps(row), flush=True)
    report["complete"] = True
    save()
    if args.smoke_only and any(row["status"] != "passed" for row in report["rows"]):
        raise RuntimeError("Marlin smoke correctness failed; see report")


if __name__ == "__main__":
    main()
