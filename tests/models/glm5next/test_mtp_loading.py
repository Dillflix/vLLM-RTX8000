# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft checkpoint configuration and owned embedding loading, without kernels."""

import pytest
import torch
from transformers import Glm5NextConfig, Glm5NextTextConfig

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.models.glm5next.common.mtp import Glm5NextMTP
from vllm.utils.torch_utils import set_default_torch_dtype


def _embedding_only_hf_override(config):
    # Keep the speculative scheduling budget valid while constructing no
    # decoder layers in this configuration/weight-loader unit test.
    config.n_predict = 1
    return config


def _write_config(path, hidden_size, group_size):
    text = Glm5NextTextConfig(
        vocab_size=65,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=0,
        num_nextn_predict_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        layer_types=[],
        index_topk=128,
        index_kpool=4,
        mla_use_nope=True,
        qk_rope_head_dim=0,
    )
    config = Glm5NextConfig(
        text_config=text.to_dict(),
        architectures=["Glm5NextForConditionalGeneration"],
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "ignore": ["lm_head", "re:.*embed_tokens.*"],
            "config_groups": {
                "experts": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": group_size,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": None,
                }
            },
        },
    )
    config.save_pretrained(path)


@pytest.fixture
def mtp_configs(tmp_path):
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    _write_config(target_path, hidden_size=8, group_size=64)
    _write_config(draft_path, hidden_size=16, group_size=128)
    target = ModelConfig(
        model=str(target_path),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=128,
        enforce_eager=True,
        hf_overrides=_embedding_only_hf_override,
    )
    parallel = ParallelConfig()
    speculation = SpeculativeConfig(
        model=str(draft_path),
        method="mtp",
        num_speculative_tokens=1,
        target_model_config=target,
        target_parallel_config=parallel,
    )
    return VllmConfig(
        model_config=target,
        cache_config=CacheConfig(enable_prefix_caching=False, mamba_block_size=128),
        parallel_config=parallel,
        speculative_config=speculation,
        additional_config={"unrelated": "retained"},
    )


def test_mtp_constructor_uses_draft_config_without_mutating_target(
    mtp_configs, dist_init
):
    target_model = mtp_configs.model_config
    target_quant = mtp_configs.quant_config
    additional = mtp_configs.additional_config
    draft = mtp_configs.speculative_config.draft_model_config
    # Zero decoder layers exercise the real constructor/configuration/embedding
    # path without allocating attention caches or invoking a CUDA kernel.
    with (
        set_current_vllm_config(mtp_configs),
        set_default_torch_dtype(mtp_configs.model_config.dtype),
        torch.device("cpu"),
    ):
        model = Glm5NextMTP(vllm_config=mtp_configs)
    assert model.config is draft.hf_text_config
    assert model.config.hidden_size == 16
    assert model.model.embed_tokens.weight.shape[1] == 16
    assert isinstance(model.quant_config, compressed_tensors.CompressedTensorsConfig)
    assert model.quant_config is not target_quant
    assert model.quant_config.target_scheme_map["Linear"]["weights"].group_size == 128
    assert target_quant.target_scheme_map["Linear"]["weights"].group_size == 64
    assert mtp_configs.model_config is target_model
    assert mtp_configs.quant_config is target_quant
    assert mtp_configs.additional_config is additional
    assert mtp_configs.additional_config == {"unrelated": "retained"}


@pytest.mark.parametrize("prefix", ["model.", "model.language_model."])
def test_mtp_loads_base_embedding_and_zeroes_padding(mtp_configs, dist_init, prefix):
    with (
        set_current_vllm_config(mtp_configs),
        set_default_torch_dtype(mtp_configs.model_config.dtype),
        torch.device("cpu"),
    ):
        model = Glm5NextMTP(vllm_config=mtp_configs)
        weights = torch.arange(65 * 16).reshape(65, 16).bfloat16()
        loaded = model.load_weights([(prefix + "embed_tokens.weight", weights)])
    assert model.has_own_embed_tokens
    assert loaded == {"model.embed_tokens.weight"}
    torch.testing.assert_close(
        model.model.embed_tokens.weight[:65], weights.half(), rtol=0, atol=0
    )
    assert torch.count_nonzero(model.model.embed_tokens.weight[65:]) == 0


def test_mtp_rejects_missing_owned_embedding(mtp_configs, dist_init):
    with (
        set_current_vllm_config(mtp_configs),
        set_default_torch_dtype(mtp_configs.model_config.dtype),
        torch.device("cpu"),
    ):
        model = Glm5NextMTP(vllm_config=mtp_configs)
        with pytest.raises(ValueError, match="requires base embed_tokens"):
            model.load_weights([])


def test_mtp_rejects_wrong_embedding_shape(mtp_configs, dist_init):
    with (
        set_current_vllm_config(mtp_configs),
        set_default_torch_dtype(mtp_configs.model_config.dtype),
        torch.device("cpu"),
    ):
        model = Glm5NextMTP(vllm_config=mtp_configs)
        with pytest.raises((AssertionError, RuntimeError)):
            model.load_weights([("model.embed_tokens.weight", torch.zeros(65, 8))])
