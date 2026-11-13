# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP16 Tensor Core indexer logits with software E4M3 storage decoding."""

import ctypes

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.glm5next.nvidia.ops.mqa_logits_sm75 import (
    fp8_mqa_logits_sm75,
    fp8_paged_mqa_logits_sm75,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


def _inputs(rows, heads, keys):
    generator = torch.Generator(device="cpu").manual_seed(42)
    # Dyadic inputs make every dot product and weighted sum exactly representable
    # in FP32, so arithmetic and masking can be checked without loose tolerances.
    q = torch.randint(-4, 5, (rows, heads, 128), generator=generator).float()
    k = torch.randint(-4, 5, (keys, 128), generator=generator).float()
    weights = torch.randint(-2, 3, (rows, heads), generator=generator).float() / 8
    scales = torch.pow(2.0, torch.randint(-4, 1, (keys,), generator=generator))
    return q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), weights, scales


def _reference(q, k, weights, scales):
    dots = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return ((dots * scales).relu() * weights.unsqueeze(-1)).sum(1)


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("key_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("dynamic_lengths", [False, True])
def test_prefill_compressed_logits_mask_request_boundaries(
    heads, strided, query_dtype, key_dtype, dynamic_lengths
):
    q, k, weights, scales = _inputs(5, heads, 150)
    q = q.to(query_dtype)
    starts = torch.tensor([0, 67, 30, -1, 149], dtype=torch.int32)
    ends = torch.tensor([65, 140, 30, 10, 150], dtype=torch.int32)
    width = 91
    expected = torch.full((5, width), -torch.inf)
    dense = _reference(q, k, weights, scales)
    for row, (start, end) in enumerate(
        zip(starts.tolist(), ends.tolist(), strict=True)
    ):
        if start >= 0:
            expected[row, : end - start] = dense[row, start:end]
    q_gpu, k_gpu = q.cuda(), k.to(key_dtype).cuda()
    if strided:
        # Preserve true outer strides instead of silently flattening packed views.
        q_gpu = q_gpu.transpose(0, 1).contiguous().transpose(0, 1)
        k_gpu = k_gpu.t().contiguous().t()
    logits = fp8_mqa_logits_sm75(
        q_gpu,
        (k_gpu, scales.cuda()),
        weights.cuda(),
        starts.cuda(),
        ends.cuda(),
        width,
        dynamic_lengths=dynamic_lengths,
    )
    actual = logits.cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    row_ends = torch.where(starts >= 0, ends - starts, 0).cuda()
    selected = torch.empty((5, 8), dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        logits,
        torch.zeros_like(row_ends),
        row_ends,
        selected,
        5,
        logits.stride(0),
        logits.stride(1),
        8,
    )
    selected = selected.cpu()
    for row in (0, 1, 4):
        count = min(8, int(ends[row] - starts[row]))
        indices = selected[row, :count].long()
        assert (indices >= 0).all()
        assert (indices < ends[row] - starts[row]).all()
        torch.testing.assert_close(
            actual[row, indices].sort().values,
            expected[row].topk(count).values.sort().values,
            atol=0,
            rtol=0,
        )
    assert (selected[2:4] == -1).all()


@pytest.mark.skipif(
    not current_platform.is_device_capability(75),
    reason="Exact dynamic/static rounding is qualified for the SM75 compiler",
)
@pytest.mark.parametrize("width", [17, 63, 64, 65, 91])
def test_dynamic_prefill_preserves_static_rounding_at_width_boundary(width):
    """Non-dyadic scales expose contraction differences hidden by exact inputs."""
    generator = torch.Generator(device="cpu").manual_seed(713)
    count = width + 64
    q = torch.randn(5, 32, 128, generator=generator).to(torch.float8_e4m3fn).cuda()
    keys = torch.randn(count, 128, generator=generator).half().cuda()
    weights = (torch.randn(5, 32, generator=generator) / 32).cuda()
    scales = torch.rand(count, generator=generator).cuda()
    starts = torch.tensor([0, 1, -1, count - 1, count + 3], device="cuda")
    ends = torch.tensor([width, width + 1, width, count + 3, count + 9], device="cuda")
    operands = (q, (keys, scales), weights, starts, ends, width)
    static = fp8_mqa_logits_sm75(*operands)
    dynamic = fp8_mqa_logits_sm75(*operands, dynamic_lengths=True)
    lengths = torch.where(
        starts >= 0,
        torch.minimum((ends - starts).clamp(0, width), (count - starts).clamp_min(0)),
        0,
    )
    valid = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
    assert torch.equal(torch.isneginf(dynamic), ~valid)
    assert torch.isfinite(dynamic[valid]).all()
    assert torch.equal(dynamic.view(torch.int32), static.view(torch.int32))


@pytest.mark.skipif(
    not current_platform.is_device_capability(75),
    reason="Native arithmetic reproduces the pinned SM75 Triton specialization",
)
@pytest.mark.parametrize("fp8_arithmetic", [False, True])
def test_native_scoring_preserves_bits_and_exact_selection_across_graph_replays(
    fp8_arithmetic,
):
    """Non-dyadic scales expose FP8/FP16 contraction-order differences."""
    from vllm.models.glm5next.nvidia.ops import native_topk_sm75 as native

    rows, count, width = 6, 32769, 32800
    generator = torch.Generator(device="cpu").manual_seed(125)
    q = torch.randn(rows, 32, 128, generator=generator).to(torch.float8_e4m3fn).cuda()
    source = (
        torch.randn(count, 128, generator=generator)
        .to(torch.float8_e4m3fn if fp8_arithmetic else torch.float16)
        .cuda()
    )
    keys = source.half()
    weights = (torch.randn(rows, 32, generator=generator) / 32).cuda()
    scales = torch.rand(count, generator=generator).cuda()
    starts = torch.tensor(
        [0, 7, -1, count - 9, count + 1, 0], dtype=torch.int32, device="cuda"
    )
    ends = torch.tensor(
        [count, count, 9, count + 5, count + 7, 0], dtype=torch.int32, device="cuda"
    )
    scratch = torch.empty(native.scratch_elements(rows, width), device="cuda")
    selected = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    scores = torch.empty((rows, width), device="cuda")
    score = getattr(
        native.library(), "glm_score_fp8decoded" if fp8_arithmetic else "glm_score"
    )
    score.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    score.restype = ctypes.c_int

    def invoke():
        keys.copy_(source)
        native.select(
            q,
            keys,
            scales,
            weights,
            starts,
            ends,
            width,
            scratch,
            selected,
            fp8_arithmetic=fp8_arithmetic,
        )
        assert (
            score(
                *[
                    x.data_ptr()
                    for x in (q, keys, scales, weights, starts, ends, scores)
                ],
                rows,
                count,
                width,
                torch.cuda.current_stream().cuda_stream,
            )
            == 0
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for state in range(3):
        if state == 1:
            source.copy_(source.flip(0))
            weights.mul_(-1)
        if state == 2:
            ends.zero_()
        graph.replay()
        reference = fp8_mqa_logits_sm75(
            q,
            (source, scales),
            weights,
            starts,
            ends,
            width,
            dynamic_lengths=True,
        )
        assert torch.equal(scores.view(torch.int32), reference.view(torch.int32))
        expected = torch.full((rows, 512), -1, dtype=torch.int32)
        for row, (start, end) in enumerate(
            zip(starts.cpu().tolist(), ends.cpu().tolist(), strict=True)
        ):
            length = max(0, min(width, min(end, count) - start)) if start >= 0 else 0
            take = min(length, 512)
            expected[row, :take] = (
                torch.arange(take)
                if length <= 512
                else torch.argsort(
                    reference[row, :length].cpu(), descending=True, stable=True
                )[:take]
            )
        assert torch.equal(selected.cpu(), expected)


@pytest.mark.parametrize("page_size", [32, 64, 272, 1088])
@pytest.mark.parametrize("next_n", [1, 4])
@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("heads", [16, 32], ids=["heads16", "heads32"])
def test_paged_logits_use_each_query_length_and_mask_invalid_pages(
    page_size, next_n, query_dtype, heads
):
    batch, pages = 2, 6
    rows = batch * next_n
    q, k, weights, scales = _inputs(rows, heads, pages * page_size)
    q = q.to(query_dtype)
    # Extra padding between physical pages exercises unified-cache block strides.
    storage = torch.zeros((pages, page_size * 132 + 32), dtype=torch.uint8)
    flat = storage[:, : page_size * 132]
    flat[:, : page_size * 128] = k.view(torch.uint8).reshape(pages, page_size * 128)
    flat[:, page_size * 128 :].view(torch.float32).copy_(scales.view(pages, page_size))
    table = torch.tensor([[3, 1, -1], [2, 4, pages + 1]], dtype=torch.int32)
    lengths = torch.tensor(
        [
            [page_size + i + 1 for i in range(next_n)],
            [0 if i == 0 else 3 * page_size for i in range(next_n)],
        ],
        dtype=torch.int32,
    )
    width = 3 * page_size + 5
    expected = torch.full((rows, width), -torch.inf)
    k_float = k.float()
    for request in range(batch):
        for step in range(next_n):
            row = request * next_n + step
            cols = torch.arange(min(width, int(lengths[request, step])))
            physical = table[request, cols // page_size]
            valid = (physical >= 0) & (physical < pages)
            cols, physical = cols[valid], physical[valid]
            keys = physical.long() * page_size + cols % page_size
            expected[row, cols] = _reference(
                q[row : row + 1], k_float[keys], weights[row : row + 1], scales[keys]
            )[0]
    storage_gpu = storage.cuda()
    cache_gpu = storage_gpu[:, : page_size * 132].view(pages, page_size, 1, 132)
    logits = fp8_paged_mqa_logits_sm75(
        q.cuda().view(batch, next_n, heads, 128),
        cache_gpu,
        weights.cuda(),
        lengths.cuda(),
        table.cuda(),
        width,
    )
    actual = logits.cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    selected = torch.empty((rows, 8), dtype=torch.int32, device="cuda")
    ops.top_k_per_row_decode(
        logits,
        next_n,
        lengths.cuda(),
        selected,
        rows,
        logits.stride(0),
        logits.stride(1),
        8,
    )
    selected = selected.cpu()
    for row in range(rows):
        valid_count = int(torch.isfinite(expected[row]).sum())
        if valid_count:
            indices = selected[row].long()
            assert (indices >= 0).all()
            assert torch.isfinite(expected[row, indices]).all()
            torch.testing.assert_close(
                actual[row, indices].sort().values,
                expected[row].topk(8).values.sort().values,
                atol=0,
                rtol=0,
            )
        else:
            assert (selected[row] == -1).all()


@pytest.mark.parametrize("fp16_keys", [False, True])
def test_fp8_subnormal_and_extreme_values_decode_exactly(fp16_keys):
    # Opposite queries turn positive and negative finite FP8 values into ReLU
    # outputs separately; this covers all 254 finite encodings, including zeros.
    codes = torch.arange(256, dtype=torch.uint8)
    codes = codes[(codes & 127) != 127]
    keys = torch.zeros((len(codes), 128), dtype=torch.uint8)
    keys[:, 0] = codes
    q = torch.zeros((2, 16, 128))
    q[0, 0, 0], q[1, 0, 0] = 1, -1
    weights = torch.zeros((2, 16))
    weights[:, 0] = 1
    scales = torch.ones(len(codes))
    key_gpu = keys.cuda()
    if fp16_keys:
        key_gpu = key_gpu.view(torch.float8_e4m3fn).half()
    actual = fp8_mqa_logits_sm75(
        q.to(torch.float8_e4m3fn).cuda(),
        (key_gpu, scales.cuda()),
        weights.cuda(),
        torch.zeros(2, dtype=torch.int32, device="cuda"),
        torch.full((2,), len(codes), dtype=torch.int32, device="cuda"),
        len(codes),
    ).cpu()
    decoded = codes.view(torch.float8_e4m3fn).float()
    expected = torch.stack([decoded.relu(), (-decoded).relu()])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("fp16_keys", [False, True])
def test_mqa_graph_replay_reads_updated_queries(query_dtype, fp16_keys):
    q, k, weights, scales = _inputs(4, 16, 64)
    q_gpu = q.to(query_dtype).cuda()
    k_gpu, weights_gpu, scales_gpu = k.cuda(), weights.cuda(), scales.cuda()
    starts = torch.zeros(4, dtype=torch.int32, device="cuda")
    ends = torch.full((4,), 64, dtype=torch.int32, device="cuda")
    table = torch.tensor([[0, 1]] * 4, dtype=torch.int32, device="cuda")
    cache = torch.empty((2, 32 * 132), dtype=torch.uint8, device="cuda")
    cache[:, : 32 * 128] = k_gpu.view(torch.uint8).view(2, 32 * 128)
    cache[:, 32 * 128 :].view(torch.float32).copy_(scales_gpu.view(2, 32))

    def run():
        gathered_keys = k_gpu.half() if fp16_keys else k_gpu
        return (
            fp8_mqa_logits_sm75(
                q_gpu, (gathered_keys, scales_gpu), weights_gpu, starts, ends, 64
            ),
            fp8_paged_mqa_logits_sm75(
                q_gpu.view(4, 1, 16, 128),
                cache.view(2, 32, 1, 132),
                weights_gpu,
                ends.view(4, 1),
                table,
                64,
            ),
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    q_gpu.copy_((-q.float()).to(query_dtype).cuda())
    graph.replay()
    eager = run()
    for actual, expected in zip(captured, eager, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_device_capability(75),
    reason="The optional GLM prefill scratch configuration is specific to SM75",
)
@pytest.mark.parametrize("prefill_rows", [480, 508, 511, 512])
@pytest.mark.parametrize("logits_scratch", [False, True])
@pytest.mark.parametrize(
    "bounded_gather", [False, True], ids=["legacy_gather", "bounded_gather"]
)
def test_indexer_prefill_scratch_reuses_gathers_and_preserves_cache(
    query_dtype, monkeypatch, tmp_path, prefill_rows, logits_scratch, bounded_gather
):
    """Exercise profiling, lazy conversion, skipped gathers and the next request.

    Each eligible chunk is split into two logits windows; the first request begins
    with an ineligible 479-row chunk. Request two uses different physical pages.
    """
    from vllm.config import (
        CacheConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillMetadata,
        build_prefill_chunk_metadata,
    )
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        lock_workspace,
        reset_workspace_manager,
    )

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    first_rows = 479
    request0_rows = first_rows + 2 * prefill_rows
    rows, heads, keys, page = request0_rows + prefill_rows, 32, 1024, 64
    q, k, weights, scales = _inputs(rows, heads, keys * 2)
    # Avoid tied scores at the selection boundary. The CUDA top-k op returns
    # selected keys in an unspecified order, even for identical logits.
    weights = torch.randn(rows, heads, generator=torch.Generator().manual_seed(43))
    q = q.to(query_dtype).cuda()
    weights = weights.cuda()
    cache = torch.empty((32, page * 132), dtype=torch.uint8, device="cuda")
    cache[:, : page * 128] = k.cuda().view(torch.uint8).reshape(32, page * 128)
    cache[:, page * 128 :].view(torch.float32).copy_(scales.cuda().view(32, page))
    original = cache.clone()
    starts_cpu = torch.tensor([0, request0_rows, rows], dtype=torch.int32)
    starts = starts_cpu.cuda()
    lengths = torch.tensor([4096, 4096], device="cuda", dtype=torch.int32)
    compressed = lengths // 4
    table = torch.arange(32, device="cuda", dtype=torch.int32).view(2, 16)
    chunks = [
        build_prefill_chunk_metadata(
            request,
            request + 1,
            starts,
            starts_cpu,
            lengths,
            compressed,
            compressed.cpu(),
            table,
            4,
            query_slice=slice(begin, end),
            skip_kv_gather=begin > 0,
        )
        for request, begin, end in (
            (0, 0, first_rows),
            (0, first_rows, first_rows + prefill_rows),
            (0, first_rows + prefill_rows, request0_rows),
            (1, 0, prefill_rows),
        )
    ]
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=lengths,
        max_seq_len=4096,
        slot_mapping=torch.full((rows,), -1, device="cuda", dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        num_prefill_tokens=rows,
        prefill=DeepseekV32IndexerPrefillMetadata(chunks, max_prefill_seq_len=4096),
    )
    positions = torch.cat(
        (
            torch.arange(4096 - request0_rows, 4096),
            torch.arange(4096 - prefill_rows, 4096),
        )
    ).cuda()
    hidden = torch.empty((rows, 1), device="cuda", dtype=torch.float16)
    raw_keys = torch.empty((rows, 128), device="cuda", dtype=torch.float16)
    from vllm.models.glm5next.nvidia import sparse_indexer as indexer_module

    calls = []
    original_mqa = indexer_module.fp8_mqa_logits_sm75

    def observe(q, kv, *args, **kwargs):
        calls.append((q.shape[0], kv[0].dtype))
        result = original_mqa(q, kv, *args, **kwargs)
        out = kwargs.get("out")
        if logits_scratch:
            assert out is not None and result.data_ptr() == out.data_ptr()
            out_begin = out.data_ptr()
            out_end = out_begin + out.numel() * out.element_size()
            for tensor in (q, *kv, *args[:3]):
                if not tensor.numel():
                    continue
                begin = tensor.data_ptr()
                span = 1 + sum(
                    (n - 1) * s for n, s in zip(tensor.shape, tensor.stride())
                )
                end = begin + span * tensor.element_size()
                assert out_end <= begin or end <= out_begin
            reference = original_mqa(q, kv, *args, **{**kwargs, "out": None})
            assert torch.equal(result.view(torch.int32), reference.view(torch.int32))
        else:
            assert out is None
        return result

    monkeypatch.setattr(indexer_module, "fp8_mqa_logits_sm75", observe)
    model_config = None
    if bounded_gather:
        import json

        from vllm.config import ModelConfig

        model_path = tmp_path / "tiny-opt-config"
        model_path.mkdir()
        (model_path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["OPTForCausalLM"],
                    "model_type": "opt",
                    "hidden_size": 64,
                    "ffn_dim": 128,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "vocab_size": 1024,
                    "max_position_embeddings": 4096,
                    "word_embed_proj_dim": 64,
                    "torch_dtype": "float16",
                }
            )
        )
        model_config = ModelConfig(
            model=str(model_path),
            max_model_len=4096,
            dtype="half",
            skip_tokenizer_init=True,
        )
    outputs = []
    for enabled in (False, False, True):
        calls.clear()
        cfg = VllmConfig(
            **({"model_config": model_config} if model_config is not None else {}),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2,
                max_num_batched_tokens=rows,
                max_model_len=4096,
                is_encoder_decoder=False,
            ),
            cache_config=CacheConfig(block_size=64),
            additional_config={
                "glm_sm75_prefill_fp16_keys": enabled,
                "glm_sm75_mqa_scratch": logits_scratch,
                "glm_sm75_bound_gather": bounded_gather,
            },
        )
        prefix = "model.layers.0.self_attn.indexer.k_cache"
        reset_workspace_manager()
        init_workspace_manager(torch.device("cuda"))
        try:
            with set_current_vllm_config(cfg):
                cache_layer = Glm5NextIndexerCache(
                    head_dim=132,
                    dtype=torch.uint8,
                    prefix=prefix,
                    cache_config=cfg.cache_config,
                    index_kpool=4,
                )
                cache_layer.kv_cache = cache.view(32, page, 132)
                indexer = SparseAttnIndexerKpool(
                    k_cache=cache_layer,
                    quant_block_size=128,
                    scale_fmt="ue8m0",
                    topk_tokens=2048,
                    head_dim=128,
                    max_pool_len=keys,
                    max_total_seq_len=keys * 8 if bounded_gather else keys,
                    topk_indices_buffer=torch.empty(
                        rows, 2176, device="cuda", dtype=torch.int32
                    ),
                    skip_k_cache_insert=True,
                )
                assert indexer.max_total_seq_len == (
                    2 * keys if bounded_gather else keys
                )
                with set_forward_context(None, cfg):
                    indexer(hidden, q, raw_keys, weights, index_kpool=4)
                lock_workspace()
                with set_forward_context({prefix: metadata}, cfg):
                    outputs.append(
                        indexer(
                            hidden,
                            q,
                            raw_keys,
                            weights,
                            index_kpool=4,
                            positions=positions,
                        ).clone()
                    )
                if bounded_gather:
                    from dataclasses import replace

                    first_calls = list(calls)
                    combined = build_prefill_chunk_metadata(
                        0,
                        2,
                        starts,
                        starts_cpu,
                        lengths,
                        compressed,
                        compressed.cpu(),
                        table,
                        4,
                    )
                    assert combined is not None
                    assert (
                        combined.max_local_total_seq_lens == indexer.max_total_seq_len
                    )
                    combined_metadata = replace(
                        metadata,
                        prefill=DeepseekV32IndexerPrefillMetadata(
                            [combined], max_prefill_seq_len=4096
                        ),
                    )
                    with set_forward_context({prefix: combined_metadata}, cfg):
                        at_bound = indexer(
                            hidden,
                            q,
                            raw_keys,
                            weights,
                            index_kpool=4,
                            positions=positions,
                        ).clone()
                    # Combining chunks changes the key-arithmetic specialization
                    # of the first 479 rows. Compare the gather bound against a
                    # real unbounded indexer with the same combined layout.
                    assert isinstance(cfg.additional_config, dict)
                    reference_cfg = replace(
                        cfg,
                        additional_config={
                            **cfg.additional_config,
                            "glm_sm75_bound_gather": False,
                        },
                    )
                    with set_current_vllm_config(reference_cfg):
                        reference_indexer = SparseAttnIndexerKpool(
                            k_cache=cache_layer,
                            quant_block_size=128,
                            scale_fmt="ue8m0",
                            topk_tokens=2048,
                            head_dim=128,
                            max_pool_len=keys,
                            max_total_seq_len=2 * keys,
                            topk_indices_buffer=torch.empty_like(
                                indexer.topk_indices_buffer
                            ),
                            skip_k_cache_insert=True,
                        )
                    with set_forward_context(
                        {prefix: combined_metadata}, reference_cfg
                    ):
                        expected_at_bound = reference_indexer(
                            hidden,
                            q,
                            raw_keys,
                            weights,
                            index_kpool=4,
                            positions=positions,
                        ).clone()
                    torch.testing.assert_close(
                        expected_at_bound.sort(dim=-1).values,
                        at_bound.sort(dim=-1).values,
                        atol=0,
                        rtol=0,
                    )
                    calls[:] = first_calls
                    assert chunks[0] is not None
                    overflow = replace(chunks[0], max_local_total_seq_lens=2 * keys + 1)
                    invalid_metadata = replace(
                        metadata,
                        prefill=DeepseekV32IndexerPrefillMetadata(
                            [overflow], max_prefill_seq_len=4096
                        ),
                    )
                    with (
                        set_forward_context({prefix: invalid_metadata}, cfg),
                        pytest.raises(RuntimeError, match="gather exceeds"),
                    ):
                        indexer(
                            hidden,
                            q,
                            raw_keys,
                            weights,
                            index_kpool=4,
                            positions=positions,
                        )
        finally:
            reset_workspace_manager()
        baseline_windows = [
            (256, torch.float8_e4m3fn),
            (first_rows - 256, torch.float8_e4m3fn),
        ]
        eligible_type = torch.float16 if enabled else torch.float8_e4m3fn
        eligible_windows = [(256, eligible_type), (prefill_rows - 256, eligible_type)]
        assert calls == baseline_windows + eligible_windows * 3
    selected = [output.sort(dim=-1).values for output in outputs]
    torch.testing.assert_close(selected[0], selected[1], atol=0, rtol=0)
    torch.testing.assert_close(selected[0], selected[2], atol=0, rtol=0)
    assert (outputs[2][:, :2048] >= 0).all()
    torch.testing.assert_close(cache, original, atol=0, rtol=0)


@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("key_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("dynamic_lengths", [False, True])
def test_mqa_output_scratch_reuse_preserves_raw_bits_and_inputs(
    query_dtype, key_dtype, dynamic_lengths
):
    """Variable contiguous views must use their current width and overwrite poison."""
    generator = torch.Generator().manual_seed(703)
    q = torch.randn(512, 32, 128, generator=generator).to(query_dtype).cuda()
    keys = torch.randn(2305, 128, generator=generator).to(key_dtype).cuda()
    weights = torch.randn(512, 32, generator=generator).cuda()
    scales = torch.rand(2305, generator=generator).cuda()
    scratch = torch.empty(512 * 2177, dtype=torch.float32, device="cuda")
    original = [x.view(torch.uint8).clone() for x in (q, keys, weights, scales)]
    for rows, width in ((512, 2048), (508, 2177), (128, 1023), (1, 91), (512, 2048)):
        starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 4
        ends = (starts + width - torch.arange(rows, device="cuda") % 9).int()
        starts[0] = -1
        out = scratch[: rows * width].view(rows, width)
        scratch.fill_(torch.nan)
        reference = fp8_mqa_logits_sm75(
            q[:rows],
            (keys, scales),
            weights[:rows],
            starts,
            ends,
            width,
            dynamic_lengths=dynamic_lengths,
        )
        actual = fp8_mqa_logits_sm75(
            q[:rows],
            (keys, scales),
            weights[:rows],
            starts,
            ends,
            width,
            dynamic_lengths=dynamic_lengths,
            out=out,
        )
        assert actual is out
        assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
        assert not torch.isnan(actual).any()
        for tensor, before in zip((q, keys, weights, scales), original, strict=True):
            assert torch.equal(tensor.view(torch.uint8), before)


@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.float16])
@pytest.mark.parametrize("key_dtype", [torch.float8_e4m3fn, torch.float16])
def test_mqa_output_scratch_graph_replay_overwrites_changed_values(
    query_dtype, key_dtype
):
    q, keys, weights, scales = _inputs(5, 32, 128)
    q, keys = q.to(query_dtype).cuda(), keys.to(key_dtype).cuda()
    weights, scales = weights.cuda(), scales.cuda()
    starts = torch.zeros(5, device="cuda", dtype=torch.int32)
    ends = torch.full((5,), 128, device="cuda", dtype=torch.int32)
    out = torch.empty(5, 128, dtype=torch.float32, device="cuda")

    def run(scratch):
        return fp8_mqa_logits_sm75(
            q,
            (keys, scales),
            weights,
            starts,
            ends,
            128,
            dynamic_lengths=True,
            out=scratch,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(out)
    for new_length in (7, 128, 0, 91):
        q.copy_((-q.float()).to(query_dtype))
        ends.fill_(new_length)
        out.fill_(torch.nan)
        graph.replay()
        reference = run(None)
        assert captured is out
        assert torch.equal(out.view(torch.int32), reference.view(torch.int32))


@pytest.mark.parametrize("invalid", ["shape", "dtype", "strides", "device"])
def test_mqa_output_scratch_rejects_incompatible_storage(invalid):
    q, keys, weights, scales = [x.cuda() for x in _inputs(5, 32, 128)]
    starts = torch.zeros(5, device="cuda", dtype=torch.int32)
    ends = torch.full((5,), 128, device="cuda", dtype=torch.int32)
    out = {
        "shape": lambda: torch.empty(5, 127, device="cuda"),
        "dtype": lambda: torch.empty(5, 128, device="cuda", dtype=torch.float16),
        "strides": lambda: torch.empty(128, 5, device="cuda").t(),
        "device": lambda: torch.empty(5, 128),
    }[invalid]()
    with pytest.raises(ValueError, match="out must be contiguous FP32"):
        fp8_mqa_logits_sm75(q, (keys, scales), weights, starts, ends, 128, out=out)


def test_mqa_output_scratch_covers_actual_256k_profile():
    rows, count = 512, 65600
    q, keys, weights, scales = _inputs(rows, 32, count)
    q, keys, weights, scales = (
        q.cuda(),
        keys.half().cuda(),
        weights.cuda(),
        scales.cuda(),
    )
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(count - rows, count, device="cuda", dtype=torch.int32)
    out = torch.full((rows, count), torch.nan, device="cuda")
    reference = fp8_mqa_logits_sm75(
        q, (keys, scales), weights, starts, ends, count, dynamic_lengths=True
    )
    actual = fp8_mqa_logits_sm75(
        q,
        (keys, scales),
        weights,
        starts,
        ends,
        count,
        dynamic_lengths=True,
        out=out,
    )
    assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
