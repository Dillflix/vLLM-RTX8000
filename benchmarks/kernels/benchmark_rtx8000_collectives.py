# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair-local FP16 collectives: buffer sizes, chunking, and NCCL policies."""

import argparse
import datetime
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def measure(fn, group, communicator, mode):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with (
            communicator.capture() if communicator else nullcontext(),
            torch.cuda.graph(graph),
        ):
            for _ in range(20):
                fn()
    timings = []
    for _ in range(5):
        dist.barrier(group=group)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        if graph:
            graph.replay()
        else:
            for _ in range(20):
                fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / 20)
    gathered = [None, None]
    dist.all_gather_object(gathered, timings, group=group)
    return statistics.median(max(x) for x in zip(*gathered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--custom", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    os.sched_setaffinity(
        0,
        list(range(0, 10)) + list(range(20, 30))
        if local < 2
        else list(range(10, 20)) + list(range(30, 40)),
    )
    torch.set_num_threads(2)
    torch.accelerator.set_device_index(local)
    device = torch.device("cuda", local)
    dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=5))
    groups = [dist.new_group(ranks, backend="gloo") for ranks in ([0, 1], [2, 3])]
    group = groups[local // 2]
    nccl = PyNcclCommunicator(group, device)
    assert not nccl.disabled
    custom = {}
    if args.custom:
        for mib in (8, 16, 32, 64):
            custom[mib] = CustomAllreduce(group, device, max_size=mib * 1024 * 1024)
            assert not custom[mib].disabled
    report = {
        "label": args.label,
        "pair": local // 2,
        "nccl_version": nccl.nccl_version,
        "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        "rows": [],
        "complete": False,
    }
    args.output_dir.mkdir(exist_ok=True, parents=True)
    path = args.output_dir / f"{args.label}-pair{local // 2}.json"
    for byte_count in (
        5120,
        10240,
        25600,
        51200,
        102400,
        1048576,
        2621440,
        5242880,
        8388608,
        10485760,
        16777216,
        20971520,
        33554432,
    ):
        inp = (torch.arange(byte_count // 2, device=device) % 17).half() + rank + 1
        out = torch.empty_like(inp)
        expected = (torch.arange(byte_count // 2, device=device) % 17).half() * 2 + (
            (1 + 2) if local < 2 else (3 + 4)
        )
        methods = [("nccl", None, lambda inp=inp, out=out: nccl.all_reduce(inp, out))]
        for mib, comm in custom.items():
            if byte_count < mib * 1024 * 1024:

                def run(comm=comm, inp=inp, out=out):
                    comm.all_reduce(
                        inp,
                        out=out,
                        registered=torch.cuda.is_current_stream_capturing(),
                    )

                methods.append((f"custom-buffer{mib}MiB", comm, run))
        if args.custom and byte_count >= 8388608:
            comm = custom[8]

            def chunked(comm=comm, inp=inp, out=out):
                # Equal chunks under the existing 8MiB dispatch limit.
                chunk = 2621440
                for start in range(0, inp.numel(), chunk):
                    comm.all_reduce(
                        inp[start : start + chunk],
                        out=out[start : start + chunk],
                        registered=torch.cuda.is_current_stream_capturing(),
                    )

            methods.append(("custom-chunk5MiB", comm, chunked))
        for name, comm, fn in methods:
            for mode in ("eager", "graph"):
                fn()
                torch.accelerator.synchronize()
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                ms = measure(fn, group, comm, mode)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                row = {"bytes": byte_count, "method": name, "mode": mode, "ms": ms}
                report["rows"].append(row)
                if local % 2 == 0:
                    path.write_text(json.dumps(report, indent=2))
                    print(json.dumps({"pair": local // 2, **row}), flush=True)
    report["complete"] = True
    if local % 2 == 0:
        path.write_text(json.dumps(report, indent=2))
    dist.barrier()
    for comm in custom.values():
        comm.close()
    nccl.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
