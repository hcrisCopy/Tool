from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from when2tool_action.hf_agent import (
    HFCausalAgent,
    HFGenerationParameters,
    validate_qwen3_model,
)


class _Layer(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, bias=False
        )


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=2,
            hidden_size=3,
            intermediate_size=6,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(3, 6), _Layer(3, 6)])
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        suffix = torch.tensor([[7, 8]], device=kwargs["input_ids"].device)
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tools": [{"name": "tool"}],
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "PROMPT"

    def __call__(self, text, return_tensors):
        assert text == "PROMPTprefill"
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens):
        assert ids.tolist() == [7, 8]
        assert skip_special_tokens is True
        return "decoded"


def _agent() -> tuple[HFCausalAgent, _FakeQwen]:
    model = _FakeQwen().eval()
    agent = HFCausalAgent(
        model=model,
        tokenizer=_FakeTokenizer(),
        system_prompt="system",
        generation=HFGenerationParameters(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=64,
            max_model_len=1024,
        ),
        expected_architecture="Qwen3ForCausalLM",
        expected_num_hidden_layers=2,
        expected_hidden_size=3,
        expected_intermediate_size=6,
        output_normalizer=lambda text: {
            "type": "content",
            "content": text,
            "raw_text": text,
        },
    )
    return agent, model


def test_agent_is_runtime_compatible_and_uses_registered_generation_fields() -> None:
    agent, model = _agent()
    assert agent.system_prompt == "system"
    assert agent.tokenizer is not None
    with pytest.raises(RuntimeError, match="set_seed"):
        agent.generate_batch([[{"role": "user", "content": "x"}]], [[{"name": "tool"}]])

    agent.set_seed(11)
    outputs = agent.generate_batch(
        [[{"role": "user", "content": "x"}]],
        [[{"name": "tool"}]],
        prefills=["prefill"],
    )
    assert outputs == [
        {
            "type": "content",
            "content": "prefilldecoded",
            "raw_text": "prefilldecoded",
            "prompt_text": "PROMPT",
            "finish_reason": "eos",
        }
    ]
    kwargs = model.last_generate_kwargs
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["top_p"] == pytest.approx(0.8)
    assert kwargs["top_k"] == 20
    assert kwargs["repetition_penalty"] == pytest.approx(1.0)
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["pad_token_id"] == 2
    assert kwargs["eos_token_id"] == 2
    assert agent.generation_metadata["backend"] == "transformers-hf"
    assert agent.generation_metadata["seed"] == 11


def test_agent_rejects_batch_shape_mismatch_without_fallback() -> None:
    agent, _model = _agent()
    agent.set_seed(0)
    with pytest.raises(ValueError, match="length mismatch"):
        agent.generate_batch([[]], [])
    with pytest.raises(ValueError, match="prefills/messages_batch"):
        agent.generate_batch([[]], [[]], prefills=[])


def test_qwen_validation_checks_config_and_every_down_projection() -> None:
    model = _FakeQwen().eval()
    layers = validate_qwen3_model(
        model,
        expected_architecture="Qwen3ForCausalLM",
        expected_num_hidden_layers=2,
        expected_hidden_size=3,
        expected_intermediate_size=6,
    )
    assert len(layers) == 2

    model.config.intermediate_size = 7
    with pytest.raises(ValueError, match="intermediate_size"):
        validate_qwen3_model(
            model,
            expected_architecture="Qwen3ForCausalLM",
            expected_num_hidden_layers=2,
            expected_hidden_size=3,
            expected_intermediate_size=6,
        )


def test_qwen_validation_rejects_non_qwen_architecture() -> None:
    model = _FakeQwen().eval()
    model.config.architectures = ["OtherForCausalLM"]
    with pytest.raises(ValueError, match="architectures"):
        validate_qwen3_model(
            model,
            expected_architecture="Qwen3ForCausalLM",
            expected_num_hidden_layers=2,
            expected_hidden_size=3,
            expected_intermediate_size=6,
        )
