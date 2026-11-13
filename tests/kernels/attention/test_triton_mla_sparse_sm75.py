# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 NoPE sparse attention, including GLM's padded incomplete-pool tail."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_sm75 import triton_mla_sparse_attention


@pytest.mark.parametrize("tokens,topk", [(0, 128), (1, 128), (5, 2176), (16, 2176)])
@pytest.mark.parametrize("splits", [0, 1, 2, 16])
def test_fp16_sparse_mla_matches_selected_key_softmax(tokens, topk, splits):
    torch.manual_seed(127)
    device = "cuda"
    heads, dim, context = 32, 512, 513
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.float16) * 0.25
    kv = torch.randn(context, 1, dim, device=device, dtype=torch.float16) * 0.25
    indices = torch.randint(
        0, context, (tokens, 1, topk), device=device, dtype=torch.int32
    )
    indices[..., 1::3] = -1
    indices[..., 2::7] = context + 10
    if tokens > 1:
        indices[0] = -1  # Entirely masked queries must produce zero, not NaN.
        indices[-1, 0, -1] = context - 1  # Do not truncate the padded tail.

    scale = 256**-0.5
    selected = kv[:, 0][indices[:, 0].clamp(0, context - 1).long()].float()
    valid = (indices[:, 0] >= 0) & (indices[:, 0] < context)
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    probs = torch.softmax(scores, dim=-1).nan_to_num()
    expected = torch.einsum("thk,tkd->thd", probs, selected).half()
    actual = triton_mla_sparse_attention(q, kv, indices, scale, num_kv_splits=splits)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
    if tokens > 1:
        assert torch.count_nonzero(actual[0]) == 0


def test_backend_maps_requests_and_keeps_tail_with_interleaved_cache():
    from vllm.v1.attention.backends.mla.triton_mla_sparse_sm75 import (
        TritonMLASparseSM75Impl,
    )
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseMetadata

    torch.manual_seed(137)
    device = "cuda"
    tokens, heads, dim, block, width = 3, 32, 512, 64, 2176
    storage = torch.full((4, block * 2, dim), 10000.0, device=device).half()
    cache = storage[:, :block]
    cache.copy_(torch.randn_like(cache) * 0.25)
    table = torch.tensor([[2, 0], [3, 1]], device=device, dtype=torch.int32)
    request_ids = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
    indices = torch.full((tokens, width), -1, device=device, dtype=torch.int32)
    indices[:, 0] = 2
    indices[:, 1] = 65
    indices[:, -1] = 127
    q = torch.randn(tokens, heads, dim, device=device).half() * 0.25
    scale = 256**-0.5
    impl = TritonMLASparseSM75Impl(
        num_heads=heads,
        head_size=dim,
        scale=scale,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=indices,
        kv_lora_rank=512,
    )
    metadata = XPUMLASparseMetadata(
        num_reqs=2,
        max_query_len=2,
        max_seq_len=128,
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor([0, 1, 3], device=device, dtype=torch.int32),
        slot_mapping=torch.zeros(tokens, device=device, dtype=torch.int64),
        block_table=table,
        req_id_per_token=request_ids,
        block_size=block,
        topk_tokens=width,
        num_decodes=2,
        num_decode_tokens=tokens,
    )
    actual, lse = impl.forward_mqa(q, cache, metadata, layer=None)
    selected = torch.stack(
        [
            torch.stack(
                [
                    cache[table[req, index // block], index % block]
                    for index in (2, 65, 127)
                ]
            )
            for req in (0, 1, 1)
        ]
    ).float()
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    expected = torch.einsum("thk,tkd->thd", scores.softmax(-1), selected).half()
    assert lse is None
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
    empty, lse = impl.forward_mqa(q[:0], cache, metadata, layer=None)
    assert empty.shape == (0, heads, dim) and empty.dtype == q.dtype
    assert lse is None


def test_automatic_splits_keep_at_least_128_keys_per_split():
    from vllm.v1.attention.ops.triton_mla_sparse_sm75 import _choose_num_kv_splits

    assert _choose_num_kv_splits(1, 2, 2176, 72) == 16
    assert _choose_num_kv_splits(1, 2, 128, 72) == 1
    assert _choose_num_kv_splits(1, 2, 4096, 72) == 16


@pytest.mark.skipif(
    not current_platform.is_device_capability(75),
    reason="The optional GLM H32 backend configuration is specific to SM75",
)
@pytest.mark.parametrize("prefill_rows", [479, 480, 508, 511, 512])
def test_prefill_h32_backend_keeps_mixed_decode_and_physical_cache_layout(
    tmp_path, monkeypatch, prefill_rows
):
    from transformers import Glm5NextTextConfig

    from vllm.config import (
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.triton_mla_sparse_sm75 import (
        TritonMLASparseSM75Impl,
        TritonMLASparseSM75MetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        num_attention_heads=32,
        num_key_value_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_topk=2048,
        index_kpool=4,
    ).save_pretrained(tmp_path)
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=1024,
            enforce_eager=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=5,
            max_num_batched_tokens=prefill_rows + 4,
            max_model_len=1024,
            is_encoder_decoder=False,
        ),
        additional_config={"glm_sm75_prefill_h32": True},
    )
    torch.manual_seed(147)
    tokens, heads, dim, block, width = prefill_rows + 4, 32, 512, 64, 2176
    storage = torch.full((10, block * 2, dim), 10000.0, device="cuda").half()
    cache = storage[:, :block]
    cache.copy_(torch.randn_like(cache) * 0.25)
    table = torch.tensor(
        [[2, 0], [3, 1], [5, 4], [9, 6], [7, 8]], device="cuda", dtype=torch.int32
    )
    starts_cpu = torch.tensor(
        [0, 1, prefill_rows + 1, prefill_rows + 2, prefill_rows + 3, tokens],
        dtype=torch.int32,
    )
    common = CommonAttentionMetadata(
        query_start_loc=starts_cpu.cuda(),
        query_start_loc_cpu=starts_cpu,
        seq_lens=torch.tensor(
            [128, prefill_rows, 128, 128, 128], device="cuda", dtype=torch.int32
        ),
        num_reqs=5,
        num_actual_tokens=tokens,
        max_query_len=prefill_rows,
        max_seq_len=prefill_rows,
        block_table_tensor=table,
        slot_mapping=torch.zeros(tokens, device="cuda", dtype=torch.int64),
    )
    builder = TritonMLASparseSM75MetadataBuilder(
        MLAAttentionSpec(
            block_size=block, num_kv_heads=1, head_size=dim, dtype=torch.float16
        ),
        [],
        cfg,
        torch.device("cuda"),
    )
    metadata = builder.build(0, common)
    assert metadata.large_prefill_spans == (
        ((1, 1 + prefill_rows),) if prefill_rows >= 480 else ()
    )
    assert metadata.num_decode_tokens == tokens  # All routed through sparse MQA.
    indices = torch.full((tokens, width), -1, device="cuda", dtype=torch.int32)
    indices[:, 0], indices[:, 1], indices[:, -1] = 2, 65, 127
    q = torch.randn(tokens, heads, dim, device="cuda").half() * 0.25
    kwargs = dict(
        num_heads=heads,
        head_size=dim,
        scale=256**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=indices,
        kv_lora_rank=512,
    )
    reference = TritonMLASparseSM75Impl(**kwargs)
    with set_current_vllm_config(cfg):
        candidate = TritonMLASparseSM75Impl(**kwargs)
    expected, _ = reference.forward_mqa(q, cache, metadata, layer=None)
    from vllm.v1.attention.backends.mla import triton_mla_sparse_sm75 as backend

    calls = []
    original_mla = backend.triton_mla_sparse_attention

    def observe(q, *args, **kwargs):
        calls.append((q.shape[0], kwargs.get("prefill_h32", False)))
        return original_mla(q, *args, **kwargs)

    monkeypatch.setattr(backend, "triton_mla_sparse_attention", observe)
    actual, _ = candidate.forward_mqa(q, cache, metadata, layer=None)
    assert calls == (
        [(1, False), (prefill_rows, True), (3, False)]
        if prefill_rows >= 480
        else [(tokens, False)]
    )
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(actual[:1], expected[:1], atol=0, rtol=0)
    torch.testing.assert_close(
        actual[1 + prefill_rows :], expected[1 + prefill_rows :], atol=0, rtol=0
    )
    # H32 must also consume updated inputs on replay, not capture-time values.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = candidate.forward_mqa(q, cache, metadata, layer=None)
    q.neg_()
    graph.replay()
    eager, _ = candidate.forward_mqa(q, cache, metadata, layer=None)
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)


@pytest.mark.parametrize("bad_output", ["alias", "stride", "shape"])
def test_sparse_mla_output_buffer_rejects_unsafe_views(bad_output):
    q = torch.zeros(1, 32, 512, device="cuda", dtype=torch.float16)
    kv = torch.zeros(1, 1, 512, device="cuda", dtype=torch.float16)
    indices = torch.zeros(1, 1, 128, device="cuda", dtype=torch.int32)
    if bad_output == "alias":
        output = q
    elif bad_output == "stride":
        output = torch.empty(1, 32, 1024, device="cuda", dtype=q.dtype)[..., ::2]
    else:
        output = torch.empty(1, 32, 511, device="cuda", dtype=q.dtype)
    with pytest.raises(ValueError, match="non-aliasing"):
        triton_mla_sparse_attention(q, kv, indices, 0.125, out=output)
