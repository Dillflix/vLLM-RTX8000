# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare optional Turing mHC fusion with the current reference fallback.

Random four-stream H4096 inputs, FP32 projection and 20 Sinkhorn iterations.
Measures whole pre/post/fused-post-pre operations; no model or serving claim.
Neither model dispatch nor native extension selection is changed.
"""

import argparse
import hashlib
import inspect
import json
import os
from functools import partial
from pathlib import Path

import torch
from benchmark_rtx8000_marlin import measure

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.mhc.turing import (
    mhc_fused_post_pre_turing,
    mhc_post_turing,
    mhc_pre_turing,
)
from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPostOp, MHCPreOp
from vllm.triton_utils import triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 5, 512, 2048])
    args = parser.parse_args()
    torch.manual_seed(95)
    torch.set_num_threads(2)
    if torch.cuda.get_device_capability() != (7, 5):
        parser.error("This benchmark targets SM75")
    report = {
        "scope": "synthetic mHC operations; no model or serving qualification",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "source_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
            for name, obj in (("candidate", mhc_pre_turing), ("dispatch", MHCPreOp))
        },
        "projection": "unchanged FP32 cuBLAS; no FP16 projection approximation",
        "norm_rounding": (
            "FP16 contraction; normalized value rounds to weight dtype "
            "before multiplication"
        ),
        "hidden": 4096,
        "streams": 4,
        "sinkhorn_iterations": 20,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with set_current_vllm_config(VllmConfig()):
        pre, post, fused = MHCPreOp(), MHCPostOp(), MHCFusedPostPreOp()
        for tokens in args.tokens:
            residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.float16)
            x = torch.randn(tokens, 4096, device="cuda", dtype=torch.float16)
            fn = torch.randn(24, 16384, device="cuda", dtype=torch.float32) * 0.01
            scale = torch.tensor([0.2, 0.3, 0.4], device="cuda")
            base = torch.randn(24, device="cuda") * 0.1
            pm = torch.rand(tokens, 4, 1, device="cuda")
            cm = torch.rand(tokens, 4, 4, device="cuda")
            norm = torch.rand(4096, device="cuda", dtype=torch.float16)
            preargs = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
            normargs = dict(norm_weight=norm, norm_eps=1e-6)
            cases = {
                "pre": (
                    partial(pre.forward_cuda, residual, *preargs, **normargs),
                    partial(mhc_pre_turing, residual, *preargs, **normargs),
                ),
                "post": (
                    partial(post.forward_cuda, x, residual, pm, cm),
                    partial(mhc_post_turing, x, residual, pm, cm),
                ),
                "fused_post_pre": (
                    partial(
                        fused.forward_cuda, x, residual, pm, cm, *preargs, **normargs
                    ),
                    partial(
                        mhc_fused_post_pre_turing,
                        x,
                        residual,
                        pm,
                        cm,
                        *preargs,
                        **normargs,
                    ),
                ),
            }
            for name, (control, candidate) in cases.items():
                expected, actual = control(), candidate()
                if isinstance(actual, torch.Tensor):
                    actual, expected = (actual,), (expected,)
                errors = []
                for got, want in zip(actual, expected, strict=True):
                    tol = 1e-3 if got.dtype == torch.float16 else 1e-5
                    torch.testing.assert_close(got, want, atol=tol, rtol=tol)
                    errors.append((got.float() - want.float()).abs().max().item())
                row = {"tokens": tokens, "operation": name, "max_abs": errors}
                for mode in ("eager", "graph"):
                    row[mode] = {
                        "reference": measure(control, mode, 10, 5),
                        "candidate": measure(candidate, mode, 10, 5),
                    }
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(tokens, name, row["eager"]["candidate"]["median_ms"], flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
