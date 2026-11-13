# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM pipeline buffers preserve mHC streams until the receiving layer's pre-op."""

import pytest
import torch
from transformers import Glm5NextTextConfig

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.models.glm5next.common.model import Glm5NextForCausalLM


@pytest.mark.parametrize("mhc_enabled", [False, True])
def test_pipeline_buffer_and_auxiliary_view_preserve_token_layout(
    tmp_path, dist_init, mhc_enabled
):
    # Construct the real wrapper without attention layers: this tests the
    # pipeline state contract without allocating an attention cache or weights.
    hf_config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        index_topk=128,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        layer_types=[],
        mhc=mhc_enabled,
    )
    hf_config.save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=16,
            enforce_eager=True,
        )
    )
    with set_current_vllm_config(config), torch.device("cpu"):
        model = Glm5NextForCausalLM(vllm_config=config)
        received = model.make_empty_intermediate_tensors(
            3, torch.float16, torch.device("cpu")
        )
        # Distinct residual streams make premature contraction observable.
        shape = (3, 4, 8) if mhc_enabled else (3, 8)
        payload = torch.arange(torch.tensor(shape).prod()).reshape(shape).half()
        received["hidden_states"].copy_(payload)
        assert set(received.tensors) == {"hidden_states"}
        torch.testing.assert_close(received["hidden_states"], payload, rtol=0, atol=0)

        auxiliary = model.model._aux_hidden_state(
            received["hidden_states"], None, None, None
        )
        expected = payload.mean(1) if mhc_enabled else payload
        torch.testing.assert_close(auxiliary, expected, rtol=0, atol=0)
        # An auxiliary view must not mutate the stream sent to the next layer.
        torch.testing.assert_close(received["hidden_states"], payload, rtol=0, atol=0)
