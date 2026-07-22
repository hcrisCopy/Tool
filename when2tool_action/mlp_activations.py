"""Strict Qwen3 SwiGLU activation extraction for tool-action probing.

The captured feature is the input to every ``mlp.down_proj`` module.  For
Qwen3 this is exactly ``silu(gate_proj(x)) * up_proj(x)``.  Only the final
non-padding prompt token is retained.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .constants import EXPECTED_SPLIT_SIZES, SCHEMA_VERSION, UPSTREAM_COMMIT
from .hf_agent import QWEN3_4B_INTERMEDIATE_SIZE
from .io_utils import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from .runtime import EvaluationSetting, initial_messages_and_tools
from .upstream import full_menu_sha256, load_runtime


ACTIVATION_SCHEMA_VERSION = "when2tool-mlp-activations-v1"
ACTIVATION_SPLITS = ("train", "test")
DOWN_NORMS_FILENAME = "down_proj_column_norms.pt"
PINNED_TRANSFORMERS_VERSION = "4.55.2"


def runtime_provenance_identity(receipt: dict[str, Any]) -> dict[str, str]:
    """Return the exact validated receipt identity safe to embed in artifacts."""

    if not isinstance(receipt, dict):
        raise TypeError("runtime provenance receipt must be an object")
    sha256 = receipt.get("sha256")
    git_commit = receipt.get("git_commit")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("runtime provenance receipt has invalid SHA256")
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise ValueError("runtime provenance receipt has invalid git commit")
    return {
        "runtime_provenance_sha256": sha256,
        "project_git_commit": git_commit,
    }


def activation_filename(split: str) -> str:
    if split not in ACTIVATION_SPLITS:
        raise ValueError(f"Unsupported split {split!r}")
    return f"{split}_mlp_lasttoken_fulltools.pt"


def activation_manifest_filename(split: str) -> str:
    if split not in ACTIVATION_SPLITS:
        raise ValueError(f"Unsupported split {split!r}")
    return f"{split}_mlp_lasttoken_fulltools_manifest.json"


def activation_output_paths(output_dir: Path) -> tuple[Path, ...]:
    """Return all files produced by one complete extraction run."""

    return (
        output_dir / DOWN_NORMS_FILENAME,
        *(
            path
            for split in ACTIVATION_SPLITS
            for path in (
                output_dir / activation_filename(split),
                output_dir / activation_manifest_filename(split),
            )
        ),
    )


def preflight_activation_outputs(output_dir: Path) -> None:
    """Reject output collisions before model loading or GPU allocation."""

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Activation output is not a directory: {output_dir}")
    existing = sorted(output_dir.iterdir()) if output_dir.is_dir() else []
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "MLP activation output directory must be empty; found:\n" + formatted
        )


def _qwen_layers(model: Any) -> list[Any]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        raise TypeError("Expected a Qwen-like model.model.layers stack")
    result = list(layers)
    for layer_index, layer in enumerate(result):
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None)
        if down_proj is None or not hasattr(down_proj, "weight"):
            raise TypeError(f"Layer {layer_index} has no mlp.down_proj weight")
        if down_proj.weight.ndim != 2:
            raise ValueError(f"Layer {layer_index} down_proj weight is not a matrix")
    return result


def last_valid_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Validate a non-empty right-padded mask and return final token indices."""

    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("attention_mask must have shape [batch, positive_sequence]")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must contain only 0/1 values")
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("Every prompt must contain at least one non-padding token")
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    expected = positions < lengths.unsqueeze(1)
    if not torch.equal(attention_mask.to(dtype=torch.bool), expected):
        raise ValueError("Only contiguous right padding is supported")
    return lengths - 1


def capture_last_token_mlp_activations(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Capture ``down_proj`` inputs as a CPU float16 ``[B, L, I]`` tensor."""

    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have equal [B, S] shape")
    layers = _qwen_layers(model)
    last_indices = last_valid_token_indices(attention_mask)
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles: list[Any] = []

    def make_hook(layer_index: int) -> Callable[[Any, tuple[Any, ...]], None]:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if captured[layer_index] is not None:
                raise RuntimeError(f"Layer {layer_index} down_proj ran more than once")
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(
                    f"Layer {layer_index} down_proj must receive one tensor input"
                )
            activation = inputs[0]
            if activation.ndim != 3:
                raise ValueError(
                    f"Layer {layer_index} down_proj input must be [B,S,I]"
                )
            if tuple(activation.shape[:2]) != tuple(input_ids.shape):
                raise ValueError(
                    f"Layer {layer_index} down_proj input batch/sequence mismatch"
                )
            batch_indices = torch.arange(
                activation.shape[0], device=activation.device
            )
            selected = activation[
                batch_indices, last_indices.to(device=activation.device), :
            ]
            stored = selected.detach().to(
                device="cpu", dtype=torch.float16
            )
            if not torch.all(torch.isfinite(stored)):
                raise ValueError(
                    f"Layer {layer_index} activation is non-finite after float16 storage conversion"
                )
            captured[layer_index] = stored

        return hook

    try:
        for layer_index, layer in enumerate(layers):
            handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_index))
            )
        backbone = model.model
        with torch.inference_mode():
            backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index, value in enumerate(captured) if value is None]
    if missing:
        raise RuntimeError(f"down_proj hooks did not run for layers {missing}")
    tensors = [value for value in captured if value is not None]
    intermediate_sizes = {int(value.shape[1]) for value in tensors}
    if len(intermediate_sizes) != 1:
        raise ValueError(
            f"All layers must share one intermediate size, got {intermediate_sizes}"
        )
    return torch.stack(tensors, dim=1).contiguous()


def down_proj_column_norms(model: Any) -> torch.Tensor:
    """Return ``||W_down[:, i]||_2`` for every layer/neuron on CPU."""

    rows: list[torch.Tensor] = []
    for layer_index, layer in enumerate(_qwen_layers(model)):
        weight = layer.mlp.down_proj.weight.detach()
        norms = torch.linalg.vector_norm(weight.to(dtype=torch.float32), dim=0)
        if not torch.all(torch.isfinite(norms)) or torch.any(norms <= 0):
            raise ValueError(f"Layer {layer_index} has invalid down_proj column norms")
        rows.append(norms.cpu())
    sizes = {int(row.numel()) for row in rows}
    if len(sizes) != 1:
        raise ValueError(f"All layers must share one intermediate size, got {sizes}")
    return torch.stack(rows, dim=0).contiguous()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_qwen3(
    config: ExperimentConfig, device: str
) -> tuple[Any, Any]:
    """Load only local Qwen3 files; network and remote code are forbidden."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    import transformers
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "MLP extraction requires transformers=="
            f"{PINNED_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.paths.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not isinstance(getattr(tokenizer, "chat_template", None), str) or not tokenizer.chat_template:
        raise ValueError("The local Qwen3 tokenizer must define chat_template")
    if tokenizer.eos_token_id is None:
        raise ValueError("The local Qwen3 tokenizer must define eos_token_id")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("The local Qwen3 tokenizer must define pad_token_id")
    model = Qwen3ForCausalLM.from_pretrained(
        config.paths.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    if not isinstance(model, Qwen3ForCausalLM):
        raise TypeError(f"Expected Qwen3ForCausalLM, got {type(model).__name__}")
    if model.config.architectures != [config.model.architecture]:
        raise ValueError(f"Unexpected model architecture {model.config.architectures}")
    if int(model.config.num_hidden_layers) != config.model.num_hidden_layers:
        raise ValueError("Unexpected Qwen3 layer count")
    if int(model.config.hidden_size) != config.model.hidden_size:
        raise ValueError("Unexpected Qwen3 hidden size")
    if model.config._attn_implementation != "sdpa":
        raise ValueError("MLP extraction requires the memory-efficient SDPA backend")
    model.config.use_cache = False
    layers = _qwen_layers(model)
    if len(layers) != config.model.num_hidden_layers:
        raise ValueError("Qwen3 layer stack does not match model config")
    expected_intermediate = int(model.config.intermediate_size)
    if expected_intermediate != QWEN3_4B_INTERMEDIATE_SIZE:
        raise ValueError(
            "Unexpected Qwen3-4B intermediate size: "
            f"{expected_intermediate} != {QWEN3_4B_INTERMEDIATE_SIZE}"
        )
    if any(layer.mlp.down_proj.weight.shape[1] != expected_intermediate for layer in layers):
        raise ValueError("Qwen3 down_proj dimensions do not match intermediate_size")
    return model, tokenizer


def _load_label_rows(
    path: Path,
    *,
    split: str,
    model_slug: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not isinstance(artifact.get("rows"), list):
        raise TypeError(f"{path} is not a label artifact")
    expected_top_level = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "split": split,
        "model": model_slug,
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
    }
    for key, expected in expected_top_level.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"{path.name} {key}={artifact.get(key)!r}, expected {expected!r}"
            )
    rows = artifact["rows"]
    if not rows:
        raise ValueError(f"{path} contains no label rows")
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            raise TypeError(f"Malformed label row in {path}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate label ID {row['id']} in {path}")
        seen.add(row["id"])
        if row.get("split") != split or row.get("gold_action") not in {
            "NONE",
            "A",
            "B",
            "C",
        }:
            raise ValueError(f"Invalid split/action metadata for label {row['id']}")
        if not isinstance(row.get("difficulty"), str) or not row["difficulty"]:
            raise ValueError(f"Label {row['id']} has invalid difficulty")
    if artifact.get("n") != len(rows):
        raise ValueError(f"{path.name} n does not match rows")
    return rows


def _render_fulltools_prompt(
    task: dict[str, Any], tokenizer: Any, system_prompt: str
) -> tuple[list[int], str, str]:
    setting = EvaluationSetting(
        name="fulltools_current_no_reasoning_mlp",
        tool_scope="full",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="off",
    )
    messages, built = initial_messages_and_tools(
        task, system_prompt=system_prompt, setting=setting
    )
    if built.menu_sha256 != full_menu_sha256():
        raise AssertionError("MLP extraction prompt does not use the canonical full menu")
    kwargs = {
        "tools": built.schemas,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if not isinstance(ids, list) or not ids or not all(isinstance(value, int) for value in ids):
        raise TypeError(f"Task {task['id']} chat template returned invalid token IDs")
    roundtrip = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids != roundtrip:
        raise ValueError(f"Task {task['id']} text/token prompt mismatch")
    return ids, hashlib.sha256(text.encode("utf-8")).hexdigest(), built.menu_sha256


def _right_pad(
    token_rows: list[list[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_rows or any(not row for row in token_rows):
        raise ValueError("Cannot collate an empty prompt batch")
    maximum = max(len(row) for row in token_rows)
    input_ids = torch.full(
        (len(token_rows), maximum),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(token_rows):
        length = len(row)
        input_ids[row_index, :length] = torch.tensor(
            row, dtype=torch.long, device=device
        )
        attention_mask[row_index, :length] = 1
    last_valid_token_indices(attention_mask)
    return input_ids, attention_mask


def extract_activation_split(
    tasks: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    split: str,
    task_path: Path,
    label_path: Path,
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    down_norms: torch.Tensor,
    down_norms_sha256: str,
    output_dir: Path,
    batch_size: int,
    runtime_provenance: dict[str, str],
) -> tuple[Path, Path]:
    """Extract and atomically save one split without retaining model outputs."""

    if split not in ACTIVATION_SPLITS:
        raise ValueError(f"Unsupported split {split!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not task_path.is_file() or not label_path.is_file():
        missing = task_path if not task_path.is_file() else label_path
        raise FileNotFoundError(missing)
    labels = {row["id"]: row for row in label_rows}
    task_ids = [task["id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(labels):
        raise ValueError(f"{split}: task/label ID sets differ")
    for task in tasks:
        label = labels[task["id"]]
        if task.get("difficulty") != label["difficulty"]:
            raise ValueError(f"Task {task['id']} difficulty differs from its label")
        if task.get("tool_scope") != "full":
            raise ValueError(f"Task {task['id']} is not a full-tools task")

    layers = _qwen_layers(model)
    n_layers = len(layers)
    intermediate_size = int(layers[0].mlp.down_proj.weight.shape[1])
    if tuple(down_norms.shape) != (n_layers, intermediate_size):
        raise ValueError("down_proj norm tensor does not match the model")
    activations = torch.empty(
        (len(tasks), n_layers, intermediate_size), dtype=torch.float16
    )
    utils, _, _ = load_runtime()
    system_prompt = utils.get_system_prompt(
        utils.detect_tool_format(str(config.paths.model))
    )
    device = next(model.parameters()).device
    metadata: list[dict[str, Any]] = []

    progress = tqdm(
        range(0, len(tasks), batch_size),
        desc=f"MLP activations {split}",
        unit="batch",
    )
    for start in progress:
        batch_tasks = tasks[start : start + batch_size]
        rendered = [
            _render_fulltools_prompt(task, tokenizer, system_prompt)
            for task in batch_tasks
        ]
        token_rows = [row[0] for row in rendered]
        for task, ids in zip(batch_tasks, token_rows, strict=True):
            if len(ids) > config.generation.max_model_len:
                raise ValueError(f"Task {task['id']} prompt is too long: {len(ids)}")
        input_ids, attention_mask = _right_pad(
            token_rows, tokenizer.pad_token_id, device
        )
        captured = capture_last_token_mlp_activations(
            model, input_ids, attention_mask
        )
        stop = start + len(batch_tasks)
        if tuple(captured.shape) != (
            len(batch_tasks),
            n_layers,
            intermediate_size,
        ):
            raise AssertionError(f"Unexpected captured shape {tuple(captured.shape)}")
        activations[start:stop].copy_(captured)
        for task, render, label in zip(
            batch_tasks,
            rendered,
            (labels[task["id"]] for task in batch_tasks),
            strict=True,
        ):
            ids, prompt_sha256, menu_sha256 = render
            metadata.append(
                {
                    "id": task["id"],
                    "difficulty": label["difficulty"],
                    "gold_action": label["gold_action"],
                    "prompt_sha256": prompt_sha256,
                    "menu_sha256": menu_sha256,
                    "input_tokens": len(ids),
                    "decision_index": len(ids) - 1,
                    "decision_token_id": ids[-1],
                    "decision_token_text": tokenizer.decode([ids[-1]]),
                }
            )

    expected = (len(tasks), n_layers, intermediate_size)
    if tuple(activations.shape) != expected or len(metadata) != len(tasks):
        raise AssertionError(f"{split}: incomplete activation extraction")
    activation_path = output_dir / activation_filename(split)
    manifest_path = output_dir / activation_manifest_filename(split)
    atomic_torch_save(activation_path, activations)
    activation_sha256 = sha256_file(activation_path)
    manifest = {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "split": split,
        "feature": "input to Qwen3 mlp.down_proj at final non-padding prompt token",
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
        "right_padding": True,
        "attention_implementation": model.config._attn_implementation,
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "model": {
            "slug": config.model.slug,
            "architecture": config.model.architecture,
            "num_hidden_layers": n_layers,
            "hidden_size": config.model.hidden_size,
            "intermediate_size": intermediate_size,
        },
        "shape": list(activations.shape),
        "dtype": str(activations.dtype),
        "tensor_file": activation_path.name,
        "tensor_sha256": activation_sha256,
        "down_proj_column_norms_file": DOWN_NORMS_FILENAME,
        "down_proj_column_norms_shape": list(down_norms.shape),
        "down_proj_column_norms_dtype": str(down_norms.dtype),
        "down_proj_column_norms_sha256": down_norms_sha256,
        "config_sha256": sha256_file(config.source),
        "model_config_sha256": sha256_file(config.paths.model / "config.json"),
        "tasks_sha256": sha256_file(task_path),
        "labels_sha256": sha256_file(label_path),
        "ids": task_ids,
        "ids_sha256": canonical_json_sha256(task_ids),
        "batch_size": batch_size,
        **runtime_provenance,
        "task_meta": metadata,
    }
    atomic_write_json(manifest_path, manifest)
    return activation_path, manifest_path


def extract_all_activations(
    tasks_by_split: dict[str, list[dict[str, Any]]],
    *,
    task_paths: dict[str, Path],
    label_paths: dict[str, Path],
    config: ExperimentConfig,
    output_dir: Path,
    batch_size: int,
    device: str,
    runtime_provenance: dict[str, Any],
) -> None:
    """Extract train/test MLP features after one global collision preflight."""

    preflight_activation_outputs(output_dir)
    runtime_identity = runtime_provenance_identity(runtime_provenance)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for split in ACTIVATION_SPLITS:
        if split not in tasks_by_split or split not in task_paths or split not in label_paths:
            raise KeyError(f"Missing {split} extraction input")
        if not task_paths[split].is_file():
            raise FileNotFoundError(task_paths[split])
        if not label_paths[split].is_file():
            raise FileNotFoundError(label_paths[split])
        if len(tasks_by_split[split]) != EXPECTED_SPLIT_SIZES[split]:
            raise ValueError(
                f"{split} Stage-5 extraction requires exactly "
                f"{EXPECTED_SPLIT_SIZES[split]} tasks, got "
                f"{len(tasks_by_split[split])}"
            )
    label_rows = {
        split: _load_label_rows(
            label_paths[split], split=split, model_slug=config.model.slug
        )
        for split in ACTIVATION_SPLITS
    }
    _seed_everything(config.generation.seeds[0])
    model, tokenizer = _load_qwen3(config, device)
    norms = down_proj_column_norms(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    norms_path = output_dir / DOWN_NORMS_FILENAME
    atomic_torch_save(norms_path, norms)
    norms_sha256 = sha256_file(norms_path)
    for split in ACTIVATION_SPLITS:
        extract_activation_split(
            tasks_by_split[split],
            label_rows[split],
            split=split,
            task_path=task_paths[split],
            label_path=label_paths[split],
            config=config,
            model=model,
            tokenizer=tokenizer,
            down_norms=norms,
            down_norms_sha256=norms_sha256,
            output_dir=output_dir,
            batch_size=batch_size,
            runtime_provenance=runtime_identity,
        )
