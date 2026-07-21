"""Architecture-faithful Qwen block-hidden extraction at the decision token."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .prompting import initial_messages, render_prompt, render_prompt_ids, tools_for_variant


@dataclass(frozen=True)
class EncodedPrompt:
    original_index: int
    task_id: int
    input_ids: list[int]
    prompt_hash: str
    token_count: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model(config: ExperimentConfig) -> tuple[Any, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.paths.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.paths.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    if model.config.architectures != [config.model.architecture]:
        raise ValueError(
            f"Architecture mismatch: {model.config.architectures} != "
            f"{[config.model.architecture]}"
        )
    if model.config.num_hidden_layers != config.model.num_hidden_layers:
        raise ValueError("Unexpected number of decoder layers")
    if model.config.hidden_size != config.model.hidden_size:
        raise ValueError("Unexpected hidden size")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise TypeError("Model parameters are not BF16")
    return model, tokenizer


def encode_prompts(
    tasks: list[dict[str, Any]], tokenizer: Any, variant: str, max_length: int
) -> list[EncodedPrompt]:
    encoded: list[EncodedPrompt] = []
    for index, task in enumerate(
        tqdm(tasks, desc=f"render {variant}", unit="prompt")
    ):
        messages = initial_messages(task, no_tool=False)
        tools = tools_for_variant(task, variant)
        rendered = render_prompt(tokenizer, messages, tools)
        input_ids = render_prompt_ids(tokenizer, messages, tools)
        roundtrip_ids = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if input_ids != roundtrip_ids:
            raise ValueError(
                f"Task {task['id']} chat-template text/token rendering mismatch"
            )
        if len(input_ids) > max_length:
            raise ValueError(
                f"Task {task['id']} prompt has {len(input_ids)} tokens, exceeding "
                f"max_model_len={max_length}"
            )
        encoded.append(
            EncodedPrompt(
                original_index=index,
                task_id=task["id"],
                input_ids=input_ids,
                prompt_hash=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                token_count=len(input_ids),
            )
        )
    return encoded


class BlockHiddenCapture:
    """Capture h_0 and raw h_1..h_L without changing model outputs."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.layers = model.model.layers
        self.handles: list[Any] = []
        self.decision_indices: torch.Tensor | None = None
        self.captured: list[torch.Tensor | None] = [None] * (len(self.layers) + 1)

    def _select(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[-1] != self.model.config.hidden_size:
            raise ValueError(f"Unexpected block hidden shape: {tuple(tensor.shape)}")
        if self.decision_indices is None:
            raise RuntimeError("Decision indices were not set before model forward")
        if tensor.shape[0] != self.decision_indices.shape[0]:
            raise ValueError("Hidden batch size does not match decision-index batch")
        batch_indices = torch.arange(tensor.shape[0], device=tensor.device)
        return tensor[batch_indices, self.decision_indices, :].detach().clone()

    def _embedding_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError("First decoder block did not receive a hidden-state tensor")
        self.captured[0] = self._select(args[0])

    def _layer_hook(self, layer_index: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Decoder block {layer_index + 1} returned {type(output).__name__}"
                )
            self.captured[layer_index + 1] = self._select(output)

        return hook

    def __enter__(self) -> "BlockHiddenCapture":
        self.handles.append(
            self.layers[0].register_forward_pre_hook(self._embedding_hook)
        )
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self, decision_indices: torch.Tensor) -> None:
        self.decision_indices = decision_indices
        self.captured = [None] * (len(self.layers) + 1)

    def stacked(self) -> torch.Tensor:
        missing = [index for index, value in enumerate(self.captured) if value is None]
        if missing:
            raise RuntimeError(f"Missing hidden captures at indices: {missing}")
        values = [value for value in self.captured if value is not None]
        return torch.stack(values, dim=1).float().cpu()


def _pad_batch(tokenizer: Any, records: list[EncodedPrompt]) -> dict[str, torch.Tensor]:
    padded = tokenizer.pad(
        {"input_ids": [record.input_ids for record in records]},
        padding=True,
        return_tensors="pt",
    )
    attention_mask = padded["attention_mask"].long()
    positions = torch.arange(attention_mask.shape[1]).expand_as(attention_mask)
    decision_indices = positions.masked_fill(attention_mask == 0, -1).amax(dim=1)
    expected = torch.tensor([record.token_count - 1 for record in records])
    if not torch.equal(decision_indices.cpu(), expected):
        raise AssertionError("Decision-token positions do not match unpadded lengths")
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "decision_indices": decision_indices,
    }


def _forward_backbone(
    model: Any,
    capture: BlockHiddenCapture,
    batch: dict[str, torch.Tensor],
    *,
    output_hidden_states: bool,
) -> tuple[torch.Tensor, Any]:
    device = next(model.parameters()).device
    decision_indices = batch["decision_indices"].to(device)
    capture.reset(decision_indices)
    with torch.inference_mode():
        output = model.model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    return capture.stacked(), output


def validate_hidden_semantics(
    model: Any, tokenizer: Any, capture: BlockHiddenCapture, record: EncodedPrompt
) -> dict[str, float]:
    batch = _pad_batch(tokenizer, [record])
    captured, public_output = _forward_backbone(
        model, capture, batch, output_hidden_states=True
    )
    public = public_output.hidden_states
    if public is None or len(public) != model.config.num_hidden_layers + 1:
        raise AssertionError("Unexpected public hidden-state tuple length")
    decision_index = int(batch["decision_indices"][0])
    public_pooled = torch.stack(
        [state[0, decision_index, :].float().cpu() for state in public], dim=0
    )
    torch.testing.assert_close(captured[0, :36], public_pooled[:36], rtol=0, atol=0)
    raw_last = captured[0, 36].to(next(model.parameters()).device).to(torch.bfloat16)
    normalized_last = model.model.norm(raw_last).float().cpu()
    torch.testing.assert_close(normalized_last, public_pooled[36], rtol=0, atol=0)
    residual = captured[:, 1:] - captured[:, :-1]
    reconstructed = captured[:, :1] + residual.cumsum(dim=1)
    max_error = float((reconstructed - captured[:, 1:]).abs().max())
    if max_error > 1e-4:
        raise AssertionError(f"Residual telescoping error is too large: {max_error}")
    return {
        "public_prefix_max_abs_error": float(
            (captured[0, :36] - public_pooled[:36]).abs().max()
        ),
        "final_norm_max_abs_error": float(
            (normalized_last - public_pooled[36]).abs().max()
        ),
        "residual_reconstruction_max_abs_error": max_error,
    }


def extract_block_hidden(
    tasks: list[dict[str, Any]], config: ExperimentConfig, variant: str
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    if not tasks:
        raise ValueError("No tasks supplied for hidden extraction")
    _seed_everything(config.analysis.seed)
    model, tokenizer = _load_model(config)
    encoded = encode_prompts(
        tasks, tokenizer, variant, config.generation.max_model_len
    )
    ordered = sorted(encoded, key=lambda record: (record.token_count, record.task_id))
    output_by_index: dict[int, torch.Tensor] = {}
    metadata_by_index: dict[int, dict[str, Any]] = {}

    with BlockHiddenCapture(model) as capture:
        validation = validate_hidden_semantics(model, tokenizer, capture, ordered[0])
        for start in tqdm(
            range(0, len(ordered), config.extraction_batch_size),
            desc=f"hidden {variant}",
            unit="batch",
        ):
            records = ordered[start : start + config.extraction_batch_size]
            batch = _pad_batch(tokenizer, records)
            hidden, _ = _forward_backbone(
                model, capture, batch, output_hidden_states=False
            )
            for row, record in enumerate(records):
                output_by_index[record.original_index] = hidden[row]
                decision_index = int(batch["decision_indices"][row])
                decision_token_id = int(batch["input_ids"][row, decision_index])
                metadata_by_index[record.original_index] = {
                    "id": record.task_id,
                    "prompt_variant": variant,
                    "prompt_hash": record.prompt_hash,
                    "input_tokens": record.token_count,
                    "decision_index": decision_index,
                    "decision_token_id": decision_token_id,
                    "decision_token_text": tokenizer.decode([decision_token_id]),
                }

    expected_indices = set(range(len(tasks)))
    if set(output_by_index) != expected_indices or set(metadata_by_index) != expected_indices:
        raise AssertionError("Hidden extraction did not preserve every input row")
    hidden_tensor = torch.stack(
        [output_by_index[index] for index in range(len(tasks))], dim=0
    )
    expected_shape = (
        len(tasks),
        config.model.num_hidden_layers + 1,
        config.model.hidden_size,
    )
    if tuple(hidden_tensor.shape) != expected_shape:
        raise AssertionError(
            f"Expected hidden shape {expected_shape}, got {tuple(hidden_tensor.shape)}"
        )
    metadata = [metadata_by_index[index] for index in range(len(tasks))]
    if [row["id"] for row in metadata] != [task["id"] for task in tasks]:
        raise AssertionError("Hidden metadata order differs from task order")
    return hidden_tensor, metadata, validation

