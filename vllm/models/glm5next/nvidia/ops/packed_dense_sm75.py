# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless attention-projection packing for the qualified SM75 profile."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from vllm import envs
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from .dense_sm75 import DenseLinear
from .packed_dense_codec import pack_mixed_tiles, unpack_mixed_tiles

SHAPES = {(4096, 4096), (4096, 8192), (12576, 4096)}
_workspaces: dict[tuple[int, int], torch.Tensor] = {}
_empty: dict[int, torch.Tensor] = {}
_loaded = False


def validate_configuration(vllm_config) -> bool:
    enabled = vllm_config.additional_config.get("glm_sm75_lossless_dense", False)
    if type(enabled) is not bool:
        raise ValueError("glm_sm75_lossless_dense must be a boolean")
    if not enabled:
        return False
    from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled
    from vllm.platforms import current_platform

    from .dense_sm75 import validate_configuration as validate_native

    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    offload = vllm_config.offload_config
    validate_native(
        dtype=model.dtype,
        capability=current_platform.get_device_capability(),
        tensor_parallel_size=parallel.tensor_parallel_size,
        pipeline_parallel_size=parallel.pipeline_parallel_size,
        model_type=model.hf_text_config.model_type,
        compilation_mode=vllm_config.compilation_config.mode,
        linear_backend=vllm_config.kernel_config.linear_backend,
        has_lora=vllm_config.lora_config is not None,
    )
    if (
        vllm_config.additional_config.get("glm_native_dense") is not True
        or model.hf_text_config.hidden_size != 4096
        or model.enable_sleep_mode
        or parallel.enable_dbo
        or envs.VLLM_BATCH_INVARIANT
        or offload.uva.cpu_offload_gb > 0
        or offload.prefetch.offload_group_size > 0
        or direct_reload_enabled()
    ):
        raise ValueError(
            "Packed GLM dense requires glm_native_dense, hidden size 4096, "
            "and no sleep, DBO, offload, reload or batch-invariant mode"
        )
    return True


def initialize(device):
    global _loaded
    if not _loaded:
        torch.ops.load_library(str(Path(__file__).with_name("libglm_packed_dense.so")))
        _loaded = True
    torch.ops.glm_packed_dense.configure(device.index)
    if device.index not in _empty:
        _empty[device.index] = torch.empty(0, device=device, dtype=torch.float32)


def dispatch(m, n, k):
    # Selected from actual GLM shards, including separate intermediate-row gates.
    if m == 1:
        return -2, 0
    if m <= 16:
        return -1, 4
    if m <= 32:
        return -1, 2
    if m <= 64:
        return (5 if n == 12576 else 4), 0
    if m <= 128:
        return (2 if n == 12576 else 5), 0
    if m <= 256:
        return 6, 0
    if n == 12576:
        return 2, 0
    if m <= 512:
        return 0, 0
    return 1, 0


def apply_packed(x, payload, metadata, n, k):
    if x.dtype != torch.float16 or x.device != payload.device:
        raise ValueError("Packed dense requires FP16 activations on the weight device")
    shape = x.shape[:-1]
    x = x.reshape(-1, k).contiguous()
    variant, splits = dispatch(x.shape[0], n, k)
    workspace = _empty[x.device.index]
    if variant == -1:
        stream = torch.cuda.current_stream(x.device)
        key = (x.device.index, stream.cuda_stream)
        workspace = _workspaces.get(key)
        # One shared buffer per execution stream, not one buffer per layer.
        # 4 * 16 rows also covers the 2 * 32-row dispatch above.
        size = 64 * 12576
        if workspace is None:
            workspace = torch.empty(size, device=x.device, dtype=torch.float32)
            _workspaces[key] = workspace
    out = torch.ops.glm_packed_dense.run(
        x, payload, metadata, workspace, n, k, variant, splits
    )
    return out.view(*shape, n)


class PackedDenseLinear(UnquantizedLinearMethod):
    supports_pre_processed_weights = False

    def apply(self, layer, x, bias=None):
        n, k = layer.glm_packed_shape
        out = apply_packed(x, layer.glm_packed_payload, layer.glm_packed_metadata, n, k)
        if bias is not None:
            out = out + bias
        return out

    def process_weights_after_loading(self, layer):
        raise RuntimeError(
            "Packed projections are installed after weight postprocessing"
        )


def pack_layers(named_layers):
    if not named_layers:
        return []
    device = named_layers[0][1].weight.device
    initialize(device)
    encoded = []
    for name, layer in named_layers:
        weight = layer.weight
        if (
            weight.device != device
            or not weight.is_contiguous()
            or weight.dtype != torch.float16
        ):
            raise ValueError("Expected contiguous CUDA FP16 weights on one device")
        cpu = weight.detach().cpu()
        words = cpu.view(torch.uint16).numpy()
        payload, metadata = pack_mixed_tiles(words)
        if not np.array_equal(
            unpack_mixed_tiles(payload, metadata, words.shape), words
        ):
            raise RuntimeError("Lossless packing roundtrip mismatch")
        proof = {
            "name": name,
            "shape": list(weight.shape),
            "source_word_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
            "original_bytes": weight.numel() * weight.element_size(),
            "packed_payload_bytes": payload.nbytes,
            "metadata_bytes": metadata.nbytes,
            "raw16_groups": int(np.count_nonzero(metadata[:, 1] == 16)),
        }
        proof["saved_bytes"] = (
            proof["original_bytes"] - payload.nbytes - metadata.nbytes
        )
        encoded.append((layer, payload, metadata, proof))
    del weight, cpu, words
    # The loader's 20-MiB split limit can keep a small packed weight in a
    # larger reused block. One bank avoids that per-projection over-allocation.
    # This is startup only; leave the loader's allocator policy unchanged.
    torch.accelerator.empty_cache()
    bank = torch.empty(
        sum(p.nbytes for _, p, _, _ in encoded), device=device, dtype=torch.uint8
    )
    meta_bank = torch.empty(
        sum(m.size for _, _, m, _ in encoded), device=device, dtype=torch.uint64
    )
    offset = meta_offset = 0
    installed = []
    for layer, data, meta, proof in encoded:
        packed = bank.narrow(0, offset, data.nbytes)
        records = meta_bank.narrow(0, meta_offset, meta.size).view(meta.shape)
        assert packed.data_ptr() % 16 == records.data_ptr() % 8 == 0
        packed.copy_(torch.from_numpy(data))
        records.copy_(torch.from_numpy(meta))
        torch.cuda.current_stream(device).synchronize()
        layer.register_buffer("glm_packed_payload", packed)
        layer.register_buffer("glm_packed_metadata", records)
        layer.glm_packed_shape = tuple(proof["shape"])
        layer.register_parameter("weight", None)
        layer.quant_method = PackedDenseLinear()
        installed.append(proof)
        offset += data.nbytes
        meta_offset += meta.size
    assert offset == bank.numel() and meta_offset == meta_bank.numel()
    return installed


def install(model):
    if envs.VLLM_BATCH_INVARIANT:
        raise ValueError("Packed dense has changed small-batch reduction order")
    named_layers = []
    for name, layer in model.named_modules():
        weight = getattr(layer, "weight", None)
        if not (
            name.startswith("model.layers.")
            and name.endswith((".self_attn.in_proj_qkvbfg_a", ".self_attn.o_proj"))
            and isinstance(layer, LinearBase)
            and type(getattr(layer, "quant_method", None))
            in (DenseLinear, UnquantizedLinearMethod)
            and isinstance(weight, torch.Tensor)
            and tuple(weight.shape) in SHAPES
        ):
            continue
        named_layers.append((name, layer))
    if not named_layers:
        raise ValueError("No eligible GLM attention projections found for packing")
    return pack_layers(named_layers)
