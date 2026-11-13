# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real configuration, layer and native-kernel contracts for opt-in MTP dense."""

import importlib

import pytest
import torch
from transformers import Glm5NextConfig, Glm5NextTextConfig

from vllm.config import (
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.glm5next.nvidia.ops import dense_sm75 as dense
from vllm.platforms import current_platform

sm75 = pytest.mark.skipif(
    current_platform.get_device_capability() != (7, 5),
    reason="Native RTX8000 dense kernels require SM75",
)


@pytest.fixture
def native_config_factory(tmp_path):
    text = Glm5NextTextConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        index_topk=128,
        index_kpool=4,
        mla_use_nope=True,
        qk_rope_head_dim=0,
    )
    Glm5NextConfig(
        text_config=text.to_dict(),
        architectures=["Glm5NextForConditionalGeneration"],
    ).save_pretrained(tmp_path)

    def make(*, enabled=True, base_enabled=True, draft_tokens=1, with_draft=True):
        target = ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=128,
        )
        # Validate the real TP2/PP2 configuration without launching its workers.
        # The layer tests below initialize their own genuine single-rank group.
        parallel = ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
        )
        speculative = (
            SpeculativeConfig(
                method="mtp",
                num_speculative_tokens=draft_tokens,
                target_model_config=target,
                target_parallel_config=parallel,
            )
            if with_draft
            else None
        )
        return VllmConfig(
            model_config=target,
            parallel_config=parallel,
            speculative_config=speculative,
            compilation_config=CompilationConfig(mode=0),
            additional_config={
                "glm_native_dense": base_enabled,
                "glm_native_dense_mtp": enabled,
            },
        )

    return make


def test_mtp_dense_default_is_disabled():
    assert dense.validate_mtp_configuration(VllmConfig()) is False
    assert (
        dense.validate_mtp_configuration(
            VllmConfig(additional_config={"glm_native_dense_mtp": False})
        )
        is False
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_mtp_dense_flag_requires_boolean(value):
    config = VllmConfig(additional_config={"glm_native_dense_mtp": value})
    with pytest.raises(ValueError, match="must be a boolean"):
        dense.validate_mtp_configuration(config)


@sm75
def test_mtp_dense_requires_real_qualified_configuration(native_config_factory):
    assert dense.validate_mtp_configuration(native_config_factory())
    for kwargs in (
        {"base_enabled": False},
        {"draft_tokens": 2},
        {"with_draft": False},
    ):
        with pytest.raises(ValueError, match="requires glm_native_dense and MTP1"):
            dense.validate_mtp_configuration(native_config_factory(**kwargs))


def _linear(n, k, **kwargs):
    return ReplicatedLinear(
        k, n, bias=False, params_dtype=torch.float16, disable_tp=True, **kwargs
    )


def _container(layers):
    # Real nn.Module containers give the installer its normal qualified names;
    # this component test does not impersonate a GLM model or call its methods.
    model = torch.nn.Module()
    model.model = torch.nn.ModuleDict({"layers": torch.nn.ModuleDict(layers)})
    return model


@sm75
def test_installer_preserves_real_layer_parameters_and_shared_head(
    dist_init, default_vllm_config
):
    quant = compressed_tensors.CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "weights": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 128,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": None,
                }
            },
        }
    )
    with torch.device("meta"):
        kda, mla = _linear(4096, 4096), _linear(4096, 8192)
        merged, same_shape_input = _linear(12576, 4096), _linear(4096, 4096)
        draft_output, eh_proj = _linear(4096, 8192), _linear(4096, 8192)
        quantized = _linear(4096, 8192, quant_config=quant)
        head = ParallelLMHead(
            77440, 4096, params_dtype=torch.float16, disable_tp=True, padding_size=1
        )
    originals = {id(layer): layer.quant_method for layer in (kda, mla, draft_output)}
    weights = {id(layer): layer.weight for layer in (kda, mla, draft_output)}
    target = _container(
        {
            "0": torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.ModuleDict(
                        {"o_proj": kda, "in_proj_qkvbfg_a": merged}
                    )
                }
            ),
            "3": torch.nn.ModuleDict(
                {"self_attn": torch.nn.ModuleDict({"o_proj": mla})}
            ),
            "4": torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.ModuleDict(
                        {"in_proj_qkvbfg_a": same_shape_input}
                    )
                }
            ),
        }
    )
    target.lm_head = head
    installed = dense.install(target, mtp=True)
    assert len(installed) == 5
    for layer in (kda, mla):
        shape = tuple(layer.weight.shape)
        assert (
            layer.quant_method.plans
            == dense.DENSE_PLANS[shape] | dense.MTP_TARGET_PLANS[shape]
        )
        assert layer.quant_method.original is originals[id(layer)]
        assert layer.weight is weights[id(layer)]
    assert merged.quant_method.plans == dense.DENSE_PLANS[(12576, 4096)]
    assert same_shape_input.quant_method.plans == dense.DENSE_PLANS[(4096, 4096)]
    assert isinstance(head.quant_method, dense.DenseHead)
    assert head.quant_method.plans == {1: 0}
    head_method, eh_method, quant_method = (
        head.quant_method,
        eh_proj.quant_method,
        quantized.quant_method,
    )
    draft = _container(
        {
            "45": torch.nn.ModuleDict(
                {
                    "mtp_block": torch.nn.ModuleDict(
                        {"self_attn": torch.nn.ModuleDict({"o_proj": draft_output})}
                    ),
                    "eh_proj": eh_proj,
                    "shared_head": torch.nn.ModuleDict({"head": head}),
                }
            ),
            "46": torch.nn.ModuleDict(
                {
                    "mtp_block": torch.nn.ModuleDict(
                        {"self_attn": torch.nn.ModuleDict({"o_proj": quantized})}
                    )
                }
            ),
        }
    )
    assert len(dense.install_mtp_draft(draft)) == 1
    assert draft_output.quant_method.original is originals[id(draft_output)]
    assert draft_output.quant_method.plans == dense.MTP_DRAFT_PLANS[(4096, 8192)]
    assert draft_output.weight is weights[id(draft_output)]
    assert eh_proj.quant_method is eh_method
    assert quantized.quant_method is quant_method
    assert draft.model["layers"]["45"]["shared_head"]["head"] is head
    assert head.quant_method is head_method
    assert dense.install_mtp_draft(draft) == []


@sm75
@pytest.mark.parametrize("rows", [2, 4, 6, 8])
def test_native_wrapper_and_actual_fallbacks(dist_init, default_vllm_config, rows):
    importlib.import_module("vllm._rtx8000_C")
    with torch.device("cuda"):
        layer = _linear(64, 256)
    torch.manual_seed(811 + rows)
    layer.weight.data.normal_(std=0.02)
    original = layer.quant_method
    method = dense.DenseLinear(original, {2: 0, 4: 1, 6: 1, 8: 0})
    x = torch.randn(rows, 256, dtype=torch.float16, device="cuda")
    expected = (x.float() @ layer.weight.float().T).half()
    actual = method.apply(layer, x)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    raw = torch.empty_like(actual)
    torch.ops._rtx8000_C.dense(x, layer.weight, raw, method.plans[rows])
    torch.testing.assert_close(actual, raw, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = method.apply(layer, x)
    x.neg_()
    graph.replay()
    torch.testing.assert_close(replayed, -expected, atol=5e-3, rtol=5e-3)
    for value, bias in (
        (torch.randn(7, 256, device="cuda", dtype=torch.float16), None),
        (torch.randn(10, 256, device="cuda", dtype=torch.float16), None),
        (torch.randn(rows, 512, device="cuda", dtype=torch.float16)[:, ::2], None),
        (
            torch.randn(rows * 256 + 1, device="cuda", dtype=torch.float16)[1:].view(
                rows, 256
            ),
            None,
        ),
        (x, torch.randn(64, device="cuda", dtype=torch.float16)),
        (x.float(), None),
    ):
        if value.dtype == torch.float32:
            layer = _linear(64, 256).float().cuda()
            layer.weight.data.normal_(std=0.02)
        torch.testing.assert_close(
            method.apply(layer, value, bias),
            original.apply(layer, value, bias),
            atol=0,
            rtol=0,
        )
    layer = _linear(64, 256).cuda()
    layer.weight = torch.nn.Parameter(
        torch.randn(256, 64, device="cuda", dtype=torch.float16).T,
        requires_grad=False,
    )
    assert not layer.weight.is_contiguous()
    torch.testing.assert_close(
        method.apply(layer, x), original.apply(layer, x), atol=0, rtol=0
    )
