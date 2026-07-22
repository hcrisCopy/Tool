"""Strict single-GPU Hugging Face agent used by neuron interventions.

The ordinary behavior panel uses vLLM.  Neuron hooks require access to the
PyTorch modules, so causal ablations deliberately use a separate HF backend.
Every comparison in that panel, including ``no_mask``, must use this backend.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .config import ExperimentConfig
from .upstream import load_runtime


QWEN3_4B_INTERMEDIATE_SIZE = 9728


@dataclass(frozen=True)
class HFGenerationParameters:
    """The generation fields shared with the registered behavior config."""

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int
    max_model_len: int

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_new_tokens <= 0 or self.max_model_len <= 0:
            raise ValueError("token limits must be positive")
        if self.max_new_tokens >= self.max_model_len:
            raise ValueError("max_new_tokens must be smaller than max_model_len")


def _qwen_layers(model: Any) -> list[Any]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("Expected Qwen3ForCausalLM.model.layers")
    try:
        materialized = list(layers)
    except TypeError as error:
        raise TypeError("Qwen3 model.layers is not iterable") from error
    return materialized


def validate_qwen3_model(
    model: Any,
    *,
    expected_architecture: str,
    expected_num_hidden_layers: int,
    expected_hidden_size: int,
    expected_intermediate_size: int,
) -> list[Any]:
    """Validate the exact config and module topology touched by ablation hooks."""

    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Model has no config")
    architectures = getattr(config, "architectures", None)
    if architectures != [expected_architecture]:
        raise ValueError(
            f"Unexpected model architectures {architectures!r}; "
            f"expected [{expected_architecture!r}]"
        )
    if expected_architecture != "Qwen3ForCausalLM":
        raise ValueError(
            "Neuron ablation is implemented only for Qwen3ForCausalLM, got "
            f"{expected_architecture!r}"
        )
    checks = (
        ("num_hidden_layers", expected_num_hidden_layers),
        ("hidden_size", expected_hidden_size),
        ("intermediate_size", expected_intermediate_size),
    )
    for name, expected in checks:
        actual = getattr(config, name, None)
        if actual != expected:
            raise ValueError(f"Unexpected Qwen3 {name}: {actual!r}; expected {expected}")

    layers = _qwen_layers(model)
    if len(layers) != expected_num_hidden_layers:
        raise ValueError(
            f"Qwen3 module layer count is {len(layers)}; "
            f"expected {expected_num_hidden_layers}"
        )
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None)
        if down_proj is None:
            raise TypeError(f"Layer {layer_index} has no mlp.down_proj")
        in_features = getattr(down_proj, "in_features", None)
        out_features = getattr(down_proj, "out_features", None)
        weight = getattr(down_proj, "weight", None)
        if in_features is None and weight is not None and getattr(weight, "ndim", 0) == 2:
            in_features = int(weight.shape[1])
        if out_features is None and weight is not None and getattr(weight, "ndim", 0) == 2:
            out_features = int(weight.shape[0])
        if in_features != expected_intermediate_size:
            raise ValueError(
                f"Layer {layer_index} down_proj input is {in_features!r}; "
                f"expected {expected_intermediate_size}"
            )
        if out_features != expected_hidden_size:
            raise ValueError(
                f"Layer {layer_index} down_proj output is {out_features!r}; "
                f"expected {expected_hidden_size}"
            )
    return layers


def _default_normalizer(text: str) -> dict[str, Any]:
    _utils, model_module, _registry = load_runtime()
    normalizer = getattr(model_module, "_normalize_generation_output", None)
    if not callable(normalizer):
        raise AttributeError("Pinned When2Tool model lacks output normalizer")
    normalized = normalizer(text)
    if not isinstance(normalized, dict) or normalized.get("type") not in {
        "tool",
        "content",
    }:
        raise TypeError("Pinned output normalizer returned an invalid record")
    return normalized


class HFCausalAgent:
    """Minimal AgentModel-compatible wrapper around a validated Qwen3 model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        system_prompt: str,
        generation: HFGenerationParameters,
        expected_architecture: str,
        expected_num_hidden_layers: int,
        expected_hidden_size: int,
        expected_intermediate_size: int,
        output_normalizer: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        if tokenizer is None:
            raise TypeError("tokenizer is required")
        self.layers = validate_qwen3_model(
            model,
            expected_architecture=expected_architecture,
            expected_num_hidden_layers=expected_num_hidden_layers,
            expected_hidden_size=expected_hidden_size,
            expected_intermediate_size=expected_intermediate_size,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.generation = generation
        self.max_model_len = generation.max_model_len
        self.intermediate_size = expected_intermediate_size
        self._normalizer = output_normalizer or _default_normalizer
        self._seed: int | None = None

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id if pad_token_id is None else pad_token_id

        try:
            self.device = next(model.parameters()).device
        except (AttributeError, StopIteration) as error:
            raise TypeError("Model must expose at least one parameter") from error
        if getattr(model, "training", False):
            raise ValueError("HF causal agent requires model.eval()")

    def set_seed(self, seed: int) -> None:
        """Reset all generation RNGs before one seed/condition checkpoint."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._seed = seed

    @property
    def generation_metadata(self) -> dict[str, Any]:
        return {
            "backend": "transformers-hf",
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "top_k": self.generation.top_k,
            "repetition_penalty": self.generation.repetition_penalty,
            "max_new_tokens": self.generation.max_new_tokens,
            "max_model_len": self.generation.max_model_len,
            "do_sample": self.generation.temperature > 0,
            "seed": self._seed,
            "request_seed_strategy": "sha256(generation_seed, full_prompt)",
        }

    def render_prompt(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str) or not rendered:
            raise TypeError("Tokenizer chat template did not return non-empty text")
        return rendered

    @staticmethod
    def _move_inputs(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
        if hasattr(inputs, "items"):
            moved = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        else:
            raise TypeError("Tokenizer output must be a mapping")
        input_ids = moved.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise TypeError("Tokenizer output input_ids must be a rank-2 tensor")
        return moved

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        tools_batch: list[list[dict[str, Any]]],
        prefills: list[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        if self._seed is None:
            raise RuntimeError("Call set_seed() before generation")
        if len(messages_batch) != len(tools_batch):
            raise ValueError("messages_batch/tools_batch length mismatch")
        if prefills is not None and len(prefills) != len(messages_batch):
            raise ValueError("prefills/messages_batch length mismatch")

        outputs: list[dict[str, Any]] = []
        for index, (messages, tools) in enumerate(zip(messages_batch, tools_batch)):
            prompt_text = self.render_prompt(messages, tools)
            prefill = "" if prefills is None or prefills[index] is None else prefills[index]
            if not isinstance(prefill, str):
                raise TypeError("Every prefill must be a string or None")
            full_prompt = prompt_text + prefill
            tokenized = self.tokenizer(full_prompt, return_tensors="pt")
            model_inputs = self._move_inputs(tokenized, self.device)
            prompt_length = int(model_inputs["input_ids"].shape[1])
            if prompt_length > self.generation.max_model_len - 128:
                raise ValueError(
                    f"HF prompt has {prompt_length} tokens, exceeding the registered "
                    f"context guard {self.generation.max_model_len - 128}"
                )
            generation_kwargs = {
                "do_sample": self.generation.temperature > 0,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "repetition_penalty": self.generation.repetition_penalty,
                "max_new_tokens": self.generation.max_new_tokens,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "use_cache": True,
            }
            # A global sequential RNG would couple later tasks to earlier
            # conditions (different masks can terminate at different lengths).
            # Prompt-addressed seeds keep identical requests paired while
            # allowing genuinely diverged multi-turn prompts to diverge.
            prompt_seed_payload = (
                f"{self._seed}\0{full_prompt}".encode("utf-8")
            )
            request_seed = int.from_bytes(
                hashlib.sha256(prompt_seed_payload).digest()[:8], "big"
            ) % (2**31)
            torch.manual_seed(request_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request_seed)
            with torch.inference_mode():
                generated = self.model.generate(**model_inputs, **generation_kwargs)
            sequences = getattr(generated, "sequences", generated)
            if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                raise TypeError("model.generate must return a rank-2 tensor or .sequences")
            if sequences.shape[0] != 1 or sequences.shape[1] < prompt_length:
                raise ValueError(f"Unexpected generated shape {tuple(sequences.shape)}")
            new_ids = sequences[0, prompt_length:]
            decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            if not isinstance(decoded, str):
                raise TypeError("Tokenizer decode must return a string")
            full_text = prefill + decoded
            normalized = dict(self._normalizer(full_text))
            normalized["prompt_text"] = prompt_text
            normalized["finish_reason"] = (
                "length"
                if int(new_ids.numel()) >= self.generation.max_new_tokens
                else "eos"
            )
            outputs.append(normalized)
        return outputs


def build_hf_causal_agent(
    config: ExperimentConfig,
    *,
    expected_intermediate_size: int = QWEN3_4B_INTERMEDIATE_SIZE,
    device: str = "cuda:0",
) -> HFCausalAgent:
    """Load the registered local checkpoint without remote code or downloads."""

    model_path = Path(config.paths.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if device != "cuda:0":
        raise ValueError("The registered causal protocol is single-GPU cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    utils, _model_module, _registry = load_runtime()
    tool_format = utils.detect_tool_format(str(model_path))
    if tool_format != "xml":
        raise ValueError(f"Qwen3 causal protocol requires XML tools, got {tool_format}")
    system_prompt = utils.get_system_prompt(tool_format)
    return HFCausalAgent(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        generation=HFGenerationParameters(
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            repetition_penalty=config.generation.repetition_penalty,
            max_new_tokens=config.generation.max_new_tokens,
            max_model_len=config.generation.max_model_len,
        ),
        expected_architecture=config.model.architecture,
        expected_num_hidden_layers=config.model.num_hidden_layers,
        expected_hidden_size=config.model.hidden_size,
        expected_intermediate_size=expected_intermediate_size,
    )
