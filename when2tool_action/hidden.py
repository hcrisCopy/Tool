"""All-layer residual hidden extraction at the final prompt token."""

from __future__ import annotations

import hashlib
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .io_utils import atomic_torch_save, atomic_write_json, canonical_json_sha256
from .runtime import EvaluationSetting, initial_messages_and_tools
from .upstream import full_menu_sha256, load_runtime


def _load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    import json

    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows") if isinstance(artifact, dict) else None
    if not isinstance(rows, list) or not rows:
        raise TypeError(f"{path} does not contain label rows")
    return rows


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model(config: ExperimentConfig) -> tuple[Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.paths.model, local_files_only=True, trust_remote_code=False
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
        raise ValueError(f"Unexpected model architecture {model.config.architectures}")
    if model.config.num_hidden_layers != config.model.num_hidden_layers:
        raise ValueError("Unexpected model layer count")
    if model.config.hidden_size != config.model.hidden_size:
        raise ValueError("Unexpected hidden size")
    return model, tokenizer


def _render(
    task: dict[str, Any], tokenizer: Any, system_prompt: str, tool_scope: str
) -> tuple[list[int], str, str]:
    setting = EvaluationSetting(
        name=f"{tool_scope}_current_no_reasoning_hidden",
        tool_scope=tool_scope,
        prompt_mode="current",
        require_reasoning=False,
        record_mode="off",
    )
    messages, built = initial_messages_and_tools(
        task, system_prompt=system_prompt, setting=setting
    )
    kwargs = {
        "tools": built.schemas,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    roundtrip = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids != roundtrip:
        raise ValueError(f"Task {task['id']} text/token prompt mismatch")
    return ids, hashlib.sha256(text.encode("utf-8")).hexdigest(), built.menu_sha256


def extract_split(
    tasks: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    *,
    split: str,
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    tool_scope: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    labels = {row["id"]: row for row in label_rows}
    if len(labels) != len(label_rows) or set(labels) != {task["id"] for task in tasks}:
        raise ValueError(f"{split}: task/label ID sets differ")
    utils, _, _ = load_runtime()
    system_prompt = utils.get_system_prompt(utils.detect_tool_format(str(config.paths.model)))
    hidden_rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for task in tqdm(tasks, desc=f"hidden {split}", unit="task"):
        ids, prompt_hash, menu_hash = _render(task, tokenizer, system_prompt, tool_scope)
        if tool_scope == "full" and menu_hash != full_menu_sha256():
            raise AssertionError("Hidden prompt does not use canonical full menu")
        if len(ids) > config.generation.max_model_len:
            raise ValueError(f"Task {task['id']} prompt is too long: {len(ids)}")
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != config.model.num_hidden_layers + 1:
            raise AssertionError("Unexpected public hidden-state tuple")
        pooled = torch.stack(
            [layer[0, -1, :].float().cpu() for layer in hidden_states], dim=0
        )
        hidden_rows.append(pooled)
        label = labels[task["id"]]
        metadata.append(
            {
                "id": task["id"],
                "split": split,
                "difficulty": task["difficulty"],
                "env": task["gold_env_name"],
                "category": task["category"],
                "no_tool_correct": label["no_tool_correct"],
                "tool_necessary": label["tool_necessary"],
                "gold_action": label["gold_action"],
                "prompt_hash": prompt_hash,
                "menu_sha256": menu_hash,
                "input_tokens": len(ids),
                "decision_index": len(ids) - 1,
                "decision_token_id": ids[-1],
                "decision_token_text": tokenizer.decode([ids[-1]]),
            }
        )
    hidden = torch.stack(hidden_rows, dim=0)
    expected = (len(tasks), config.model.num_hidden_layers + 1, config.model.hidden_size)
    if tuple(hidden.shape) != expected:
        raise AssertionError(f"{split}: hidden shape {tuple(hidden.shape)} != {expected}")
    hidden_path = output_dir / f"{split}_hidden_no_reasoning.pt"
    labels_path = output_dir / f"{split}_labels_no_reasoning.json"
    atomic_torch_save(hidden_path, hidden, overwrite=overwrite)
    atomic_write_json(
        labels_path,
        {
            "reasoning_mode": "no_reasoning",
            "split": split,
            "no_tool_correct": [row["no_tool_correct"] for row in metadata],
            "task_meta": metadata,
        },
        overwrite=overwrite,
    )
    atomic_write_json(
        output_dir / f"{split}_hidden_manifest.json",
        {
            "split": split,
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
            "feature": "public output_hidden_states at final input token",
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "tool_scope": tool_scope,
            "menu_sha256": full_menu_sha256() if tool_scope == "full" else None,
            "per_task_menu_hashes_sha256": canonical_json_sha256(
                [row["menu_sha256"] for row in metadata]
            ),
            "ids": [row["id"] for row in metadata],
        },
        overwrite=overwrite,
    )
    return hidden_path, labels_path


def extract_all(
    tasks_by_split: dict[str, list[dict[str, Any]]],
    label_paths: dict[str, Path],
    config: ExperimentConfig,
    output_dir: Path,
    *,
    tool_scope: str,
    overwrite: bool,
) -> None:
    _seed(config.generation.seeds[0])
    model, tokenizer = _load_model(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        extract_split(
            tasks_by_split[split],
            _load_labels(label_paths[split]),
            split=split,
            config=config,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            tool_scope=tool_scope,
            overwrite=overwrite,
        )
