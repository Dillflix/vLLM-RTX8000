# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused vocab-parallel embedding kernel
(`vllm._custom_ops.vocab_parallel_embedding`).

The guarantee: the kernel is a drop-in for the TP > 1 embedding path, i.e. it
returns bit-exactly what mask + shift + gather + `masked_fill_` returns on
each rank, and summing the per-rank outputs (what the all-reduce does)
reconstructs a full-table lookup.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
    pad_vocab_size,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused vocab-parallel embedding kernel requires CUDA or ROCm",
)


def _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size, pad=64):
    return VocabParallelEmbedding._get_indices(
        pad_vocab_size(vocab_size, pad),
        pad_vocab_size(org_vocab_size, pad),
        vocab_size,
        org_vocab_size,
        tp_rank,
        tp_size,
    )


def _reference(input_ids, weight, si):
    """The path the kernel replaces."""
    masked_input, input_mask = get_masked_input_and_mask(
        input_ids,
        si.org_vocab_start_index,
        si.org_vocab_end_index,
        si.num_org_vocab_padding,
        si.added_vocab_start_index,
        si.added_vocab_end_index,
    )
    out = torch.nn.functional.embedding(masked_input.long(), weight)
    return out.masked_fill_(input_mask.unsqueeze(-1), 0)


def _fused(input_ids, weight, si):
    return ops.vocab_parallel_embedding(
        input_ids,
        weight,
        si.org_vocab_start_index,
        si.org_vocab_end_index,
        si.num_org_vocab_padding,
        si.added_vocab_start_index,
        si.added_vocab_end_index,
    )


def _shard(full_table, si, rows_per_rank, dtype):
    """Lay a rank's rows out the way the weight loader does."""
    weight = torch.zeros(rows_per_rank, full_table.shape[1], dtype=dtype, device="cuda")
    num_org = si.org_vocab_end_index - si.org_vocab_start_index
    weight[:num_org] = full_table[si.org_vocab_start_index : si.org_vocab_end_index]
    added_at = num_org + si.num_org_vocab_padding
    num_added = si.added_vocab_end_index - si.added_vocab_start_index
    weight[added_at : added_at + num_added] = full_table[
        si.added_vocab_start_index : si.added_vocab_end_index
    ]
    return weight


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "vocab_size,org_vocab_size", [(128256, 128256), (32256, 32000)]
)
@pytest.mark.parametrize("hidden", [4096, 3584])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_matches_masked_gather(tp_size, vocab_size, org_vocab_size, hidden, id_dtype):
    """Every rank's partial output is bit-exact with the eager path, including
    the added-vocab (LoRA) rows and the ids owned by another rank."""
    set_random_seed(7)
    dtype = torch.bfloat16
    rows = pad_vocab_size(vocab_size, 64) // tp_size
    full_table = torch.randn(pad_vocab_size(vocab_size, 64), hidden, dtype=dtype).cuda()
    ids = torch.randint(0, vocab_size, (257,), dtype=id_dtype, device="cuda")

    for tp_rank in range(tp_size):
        si = _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size)
        weight = _shard(full_table, si, rows, dtype)
        torch.testing.assert_close(
            _fused(ids, weight, si), _reference(ids, weight, si), atol=0.0, rtol=0.0
        )


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("tp_size", [2, 8])
def test_rank_sum_is_full_lookup(tp_size):
    """What the all-reduce sees: the partials add up to a full-table lookup."""
    set_random_seed(11)
    vocab_size, org_vocab_size, hidden = 32256, 32000, 2048
    dtype = torch.bfloat16
    rows = pad_vocab_size(vocab_size, 64) // tp_size
    full_table = torch.randn(pad_vocab_size(vocab_size, 64), hidden, dtype=dtype).cuda()
    # Ids from both the base and the added (LoRA) range.
    ids = torch.randint(0, vocab_size, (129,), dtype=torch.int32, device="cuda")
    ids[:16] = torch.randint(
        org_vocab_size, vocab_size, (16,), dtype=torch.int32, device="cuda"
    )

    total = torch.zeros(ids.shape[0], hidden, dtype=dtype, device="cuda")
    for tp_rank in range(tp_size):
        si = _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size)
        total += _fused(ids, _shard(full_table, si, rows, dtype), si)

    torch.testing.assert_close(total, full_table[ids.long()], atol=0.0, rtol=0.0)


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_fused", [False, True])
def test_embedding_optionally_reduces_output(dtype, use_fused, default_vllm_config):
    """SP's deferred reduction preserves masked contributions and their dtype."""
    with torch.device("cuda"):
        values = torch.arange(128 * 16).reshape(128, 16)
        table = ((values % 29 - 14) / 16).to(dtype)
        ids = torch.tensor([0, 63, 64, 127, 12])
        expected = table[ids].float()
        for rank in range(2):
            local = expected.clone()
            local[ids // 64 != rank] = 0
            remote = (expected - local).to(dtype)
            group = Mock(world_size=2, rank_in_group=rank)
            # Model the other vocabulary owner's contribution, including FP8 bytes.
            group.all_reduce.side_effect = lambda output, remote=remote: (
                output + remote.view(output.dtype)
            )
            for reduce_results in (True, False):
                embedding = VocabParallelEmbedding(
                    128,
                    16,
                    params_dtype=dtype,
                    parallel_group=group,
                    reduce_results=reduce_results,
                )
                embedding.use_fused_embedding = use_fused
                embedding.weight_loader(embedding.get_parameter("weight"), table)
                actual = embedding(ids)
                assert actual.dtype == dtype
                torch.testing.assert_close(
                    actual.float(),
                    expected if reduce_results else local,
                    atol=0,
                    rtol=0,
                )


def test_glm_lossless_dictionary_preserves_all_indices_and_rejects_overflow():
    """Storage is a bit bijection, including signed zeros and NaN payloads."""
    import numpy as np

    from vllm.models.glm5next.nvidia.ops.embedding12_codec import (
        make_dictionary,
        pack_rows,
        unpack_rows,
    )

    special = np.array([0, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0xFE55], dtype=np.uint16)
    alphabet = np.concatenate(
        [special, np.setdiff1d(np.arange(4096, dtype=np.uint16) * 16, special)[:4090]]
    )
    assert np.unique(alphabet).size == 4096
    bits = np.tile(alphabet, (2, 1))
    dictionary = make_dictionary(bits)
    packed = pack_rows(bits, dictionary)
    assert packed.shape == (2, 6144)
    assert np.array_equal(unpack_rows(packed, dictionary), bits)
    with pytest.raises(ValueError, match="distinct"):
        make_dictionary(np.arange(4097, dtype=np.uint16))


def test_glm_lossless_embedding_rejects_shared_head_before_mutation():
    """Input storage cannot be compressed while the output head aliases it."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.glm5next.nvidia.ops.embedding_lossless_sm75 import (
        LosslessVocabParallelEmbedding,
        install,
    )

    with torch.device("cpu"):
        embedding = LosslessVocabParallelEmbedding(
            128, 4096, params_dtype=torch.float16, disable_tp=True
        )
        head = ParallelLMHead(128, 4096, params_dtype=torch.float16, disable_tp=True)
    model = torch.nn.Module()
    model.add_module("embed_tokens", embedding)
    model.add_module("lm_head", head)
    head.weight = embedding.weight
    original = embedding.weight
    with pytest.raises(ValueError, match="shared"):
        install(model)
    assert embedding.weight is head.weight is original
    assert not hasattr(embedding, "lossless_codes")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA lookup kernel")
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_glm_lossless_lookup_keeps_bits_and_updates_graph_ids(id_dtype):
    """Raw lookup preserves stored words and vocabulary masks during replay."""
    import numpy as np

    from vllm.models.glm5next.nvidia.ops.embedding12_codec import (
        make_dictionary,
        pack_rows,
    )
    from vllm.models.glm5next.nvidia.ops.embedding_lossless_sm75 import lookup12

    special = np.array([0, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0xFE55], dtype=np.uint16)
    alphabet = np.concatenate(
        [special, np.setdiff1d(np.arange(4096, dtype=np.uint16) * 16, special)[:4090]]
    )
    assert np.unique(alphabet).size == 4096
    indices = np.arange(8 * 4096).reshape(8, 4096)
    bits = alphabet[(indices + np.arange(8)[:, None] * 37) % len(alphabet)]
    dictionary = make_dictionary(bits)
    packed = torch.from_numpy(pack_rows(bits, dictionary)).cuda()
    words = torch.from_numpy(dictionary).cuda()
    ids = torch.tensor([[20, 21, 23], [24, 40, 42]], dtype=id_dtype, device="cuda")

    def lookup():
        # Four original rows, two padding rows, then two added vocabulary rows.
        return lookup12(ids, packed, words, 4096, 20, 24, 2, 40, 42)

    def check(output):
        expected = np.zeros((*ids.shape, 4096), dtype=np.uint16)
        for position, token in np.ndenumerate(ids.cpu().numpy()):
            if 20 <= token < 24:
                expected[position] = bits[token - 20]
            elif 40 <= token < 42:
                expected[position] = bits[token - 40 + 6]
        actual = output.cpu().view(torch.uint16).numpy()
        assert np.array_equal(actual, expected)

    check(lookup())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = lookup()
    for updated in ([[23, 22, 20], [41, -1, 40]], [[24, 20, 41], [42, 21, 23]]):
        ids.copy_(torch.tensor(updated, device="cuda", dtype=id_dtype))
        graph.replay()
        check(output)
    empty = lookup12(ids[:0], packed, words, 4096, 20, 24, 2, 40, 42)
    assert empty.shape == (0, 3, 4096)
    torch.accelerator.synchronize()
