# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise device guards, independent streams, graph replay and weight lifetime."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vllm.models.glm5next.nvidia.ops import packed_dense_sm75
from vllm.models.glm5next.nvidia.ops.packed_dense_codec import (
    pack_mixed_tiles,
    unpack_mixed_tiles,
)


def operating_snapshot():
    import subprocess

    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,power.limit,temperature.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(183331)
    lib = Path(packed_dense_sm75.__file__).with_name("libglm_packed_dense.so")
    pins = {"library": hashlib.sha256(lib.read_bytes()).hexdigest()}
    if torch.accelerator.device_count() != 4 or any(
        torch.cuda.get_device_capability(i) != (7, 5) for i in range(4)
    ):
        raise RuntimeError("This qualification requires four SM75 GPUs")
    torch.ops.load_library(str(lib))
    report = {
        "status": "RUNNING",
        "pins": pins,
        "cases": [],
        "operating_before": operating_snapshot(),
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for device in (0, 2, 1, 3, 0):
            torch.ops.glm_packed_dense.configure(device)
            with torch.accelerator.device_index(device):
                stream = torch.cuda.Stream()
                torch.cuda.set_stream(stream)
                for m in (0, 1, 3, 16, 17, 32, 64, 96, 128, 256, 511, 1024):
                    n, k = 520, 192
                    w = (torch.randn(n, k, dtype=torch.bfloat16) * 0.015625).half()
                    w[256:512] = torch.randn_like(w[256:512]) * 0.2
                    words = w.view(torch.uint16).numpy()
                    data, meta = pack_mixed_tiles(words)
                    assert set(meta[:, 1]) == {12, 16}
                    assert np.array_equal(
                        unpack_mixed_tiles(data, meta, words.shape), words
                    )
                    payload = torch.from_numpy(data).to(device)
                    metadata = torch.from_numpy(meta).to(device)
                    weight = w.to(device)
                    x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.2
                    workspace = torch.empty(
                        4 * max(1, m) * n, device=device, dtype=torch.float32
                    )
                    variants = list(range(7))
                    if m <= 512:
                        variants.append(-1)
                    if m == 1:
                        variants.append(-2)
                    stream.synchronize()
                    torch.accelerator.set_device_index((device + 1) % 4)
                    for variant in variants:
                        out = torch.ops.glm_packed_dense.run(
                            x, payload, metadata, workspace, n, k, variant, 4
                        )
                        assert (
                            torch.accelerator.current_device_index() == (device + 1) % 4
                        )
                        stream.synchronize()
                        ref = x.cpu().float() @ w.float().T
                        assert torch.allclose(
                            out.cpu().float(), ref, atol=0.001, rtol=0.005
                        )
                        assert torch.isfinite(out).all().item()
                        report["cases"].append(
                            {
                                "device": device,
                                "m": m,
                                "variant": variant,
                                "foreign_current_device_preserved": True,
                            }
                        )
                    torch.accelerator.set_device_index(device)
                    if m in (3, 96, 511):
                        variant = -1 if m == 3 else 5
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            out = torch.ops.glm_packed_dense.run(
                                x, payload, metadata, workspace, n, k, variant, 4
                            )
                            # Scratch reuse within a graph: first output must survive.
                            second = torch.ops.glm_packed_dense.run(
                                x, payload, metadata, workspace, n, k, variant, 4
                            )
                        original = x.clone()
                        for factor in (1.0, 0.25, 1.0):
                            x.copy_(original * factor)
                            graph.replay()
                            stream.synchronize()
                            ref = x.cpu().float() @ w.float().T
                            assert torch.allclose(
                                out.cpu().float(), ref, atol=0.001, rtol=0.005
                            )
                            assert torch.equal(out, second)
                        del graph, original, second
                    del out, ref, x, workspace, payload, metadata, weight
                    save()
                torch.cuda.set_stream(torch.cuda.default_stream(device))
            gc.collect()
            torch.accelerator.empty_cache()
            print("PASS_DEVICE", device, flush=True)
        report["status"] = "PASS_DEVICE_STREAM_GRAPH_COMPONENT"
    except BaseException as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        report["operating_after"] = operating_snapshot()
        save()


if __name__ == "__main__":
    main()
