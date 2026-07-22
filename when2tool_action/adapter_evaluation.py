"""Strict HF evaluation utilities for trained PEFT adapter controls."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .constants import ACTIONS, SCHEMA_VERSION, UPSTREAM_COMMIT
from .hf_agent import HFCausalAgent
from .io_utils import atomic_write_json, canonical_json_sha256, sha256_file
from .masked_lora import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MASKED_LORA_SCHEMA_VERSION,
    MAX_LENGTH,
    TRAINING_SEED,
)
from .neuron_ablation import compute_ablation_metrics


ADAPTER_EVAL_SCHEMA_VERSION = "when2tool-adapter-eval-v1"
BASE_CONDITION_ID = "base_model"
CONDITION_IDS = (
    BASE_CONDITION_ID,
    "target_neuron_lora",
    "dense_mlp_lora",
    "random_neuron_lora_seed0",
    "random_neuron_lora_seed1",
    "random_neuron_lora_seed2",
)
RANDOM_CONTROL_CONDITION_IDS = tuple(
    f"random_neuron_lora_seed{seed}" for seed in range(3)
)
H3_COMPARISON_METRICS = (
    "Accuracy",
    "TotalTC",
    "TCR",
    "ToolNeed_F1",
    "ActionAcc",
    "MacroF1_action",
    "Recall4_macro",
    "Recall_NONE",
    "Recall_A",
    "Recall_B",
    "Recall_C",
    "OverCall",
    "UnderCall",
    "WrongCat",
    "Invalid",
    "Mixed",
)
_RANDOM_CONDITION_RE = re.compile(r"random_neuron_lora_seed(?P<seed>[0-2])\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AdapterIdentity:
    condition_id: str
    kind: str
    bundle_sha256: str | None
    files: tuple[dict[str, Any], ...]
    training_control: dict[str, Any] | None
    training_protocol: dict[str, Any] | None

    def snapshot(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "bundle_sha256": self.bundle_sha256,
            "files": [dict(item) for item in self.files],
            "training_control": self.training_control,
            "training_protocol": self.training_protocol,
        }


@dataclass(frozen=True)
class CompletedEvaluationCell:
    condition_id: str
    tool_scope: str
    generation_seeds: tuple[int, ...]
    config: dict[str, Any]
    per_seed_metrics: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]
    path: Path


def validate_condition_id(condition_id: str) -> str:
    if condition_id not in CONDITION_IDS:
        raise ValueError(f"condition_id must be one of {CONDITION_IDS}")
    return condition_id


def expected_training_control(condition_id: str) -> tuple[str | None, int | None]:
    condition_id = validate_condition_id(condition_id)
    if condition_id == BASE_CONDITION_ID:
        return None, None
    if condition_id in {"target_neuron_lora", "dense_mlp_lora"}:
        return condition_id, None
    match = _RANDOM_CONDITION_RE.fullmatch(condition_id)
    if match is None:
        raise AssertionError(f"Unmapped condition ID {condition_id}")
    return "random_neuron_lora", int(match.group("seed"))


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {context} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object: {path}")
    return payload


def _adapter_files(root: Path) -> tuple[dict[str, Any], ...]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Adapter artifact may not contain symlinks: {path}")
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"Adapter directory contains no files: {root}")
    return tuple(files)


def inspect_adapter(
    condition_id: str,
    adapter_dir: Path | str | None,
    *,
    base_model_slug: str,
) -> AdapterIdentity:
    """Validate and hash either the explicit base condition or one PEFT adapter."""

    condition_id = validate_condition_id(condition_id)
    expected_control_id, expected_random_seed = expected_training_control(condition_id)
    if condition_id == BASE_CONDITION_ID:
        if adapter_dir is not None:
            raise ValueError("The base condition forbids --adapter-dir")
        return AdapterIdentity(BASE_CONDITION_ID, "base", None, (), None, None)
    if adapter_dir is None:
        raise ValueError("Non-base adapter evaluation requires --adapter-dir")
    root = Path(adapter_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    adapter_config = _read_json_object(root / "adapter_config.json", "PEFT adapter config")
    train_manifest = _read_json_object(root / "train_manifest.json", "training manifest")
    if train_manifest.get("schema_version") != MASKED_LORA_SCHEMA_VERSION:
        raise ValueError("Training manifest schema differs from masked-LoRA v1")
    if train_manifest.get("control_id") != expected_control_id:
        raise ValueError(
            f"Training control_id={train_manifest.get('control_id')!r} differs "
            f"from expected control_id={expected_control_id!r} for "
            f"condition_id={condition_id!r}"
        )
    if train_manifest.get("condition_id") != condition_id:
        raise ValueError(
            f"Training condition_id={train_manifest.get('condition_id')!r} differs "
            f"from evaluation condition_id={condition_id!r}"
        )
    if train_manifest.get("random_mask_seed") != expected_random_seed:
        raise ValueError(
            f"Training random_mask_seed={train_manifest.get('random_mask_seed')!r} "
            f"differs from condition suffix seed={expected_random_seed!r}"
        )
    if train_manifest.get("base_model") != base_model_slug:
        raise ValueError("Adapter training base model differs from evaluation base")
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("Adapter peft_type must be LORA")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Adapter task_type must be CAUSAL_LM")
    if adapter_config.get("base_model_name_or_path") != base_model_slug:
        raise ValueError(
            "Adapter base_model_name_or_path differs from the registered "
            "portable evaluation model slug"
        )
    if adapter_config.get("r") != LORA_RANK:
        raise ValueError(f"Adapter rank must be {LORA_RANK}")
    if adapter_config.get("lora_alpha") != LORA_ALPHA:
        raise ValueError(f"Adapter lora_alpha must be {LORA_ALPHA}")
    if adapter_config.get("lora_dropout") != LORA_DROPOUT:
        raise ValueError(f"Adapter lora_dropout must be {LORA_DROPOUT}")
    if adapter_config.get("bias") != "none":
        raise ValueError("Adapter bias must be 'none'")
    if adapter_config.get("inference_mode") is not True:
        raise ValueError("Saved adapter inference_mode must be true")
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list) or set(target_modules) != set(
        LORA_TARGET_MODULES
    ):
        raise ValueError(
            f"Adapter target_modules must be exactly {list(LORA_TARGET_MODULES)}"
        )
    weights = [
        path
        for path in (root / "adapter_model.safetensors", root / "adapter_model.bin")
        if path.is_file()
    ]
    if len(weights) != 1:
        raise FileNotFoundError(
            "Adapter directory must contain exactly one of adapter_model.safetensors "
            "or adapter_model.bin"
        )

    inputs = train_manifest.get("inputs")
    runtime = train_manifest.get("runtime_provenance")
    hyperparameters = train_manifest.get("hyperparameters")
    distributed = train_manifest.get("distributed")
    dependency_versions = train_manifest.get("dependency_versions")
    if not isinstance(inputs, dict):
        raise TypeError("Training manifest inputs must be an object")
    if not isinstance(runtime, dict):
        raise TypeError("Training manifest runtime_provenance must be an object")
    if not isinstance(hyperparameters, dict) or not hyperparameters:
        raise TypeError("Training manifest hyperparameters must be a non-empty object")
    if not isinstance(distributed, dict):
        raise TypeError("Training manifest distributed must be an object")
    if not isinstance(dependency_versions, dict) or not dependency_versions:
        raise TypeError("Training manifest dependency_versions must be non-empty")
    sft_jsonl = inputs.get("sft_jsonl")
    sft_manifest = inputs.get("sft_manifest")
    neuron_mask = inputs.get("neuron_mask")
    for name, value in (
        ("inputs.sft_jsonl", sft_jsonl),
        ("inputs.sft_manifest", sft_manifest),
        ("inputs.neuron_mask", neuron_mask),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"Training manifest {name} must be an object")
    hash_fields = {
        "sft_jsonl_sha256": sft_jsonl.get("sha256"),
        "sft_manifest_sha256": sft_manifest.get("sha256"),
        "sft_input_bundle_sha256": sft_manifest.get("input_bundle_sha256"),
        "neuron_mask_sha256": neuron_mask.get("sha256"),
        "runtime_provenance_sha256": runtime.get("sha256"),
        "config_sha256": train_manifest.get("config_sha256"),
        "full_menu_sha256": train_manifest.get("full_menu_sha256"),
        "protocol_fingerprint": train_manifest.get("protocol_fingerprint"),
    }
    for name, value in hash_fields.items():
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"Training manifest {name} must be a lowercase SHA256")
    project_git_commit = runtime.get("project_git_commit")
    if not isinstance(project_git_commit, str) or not project_git_commit:
        raise ValueError("Training runtime project_git_commit must be non-empty")
    world_size = distributed.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("Training distributed.world_size must be a positive integer")
    if 8 % world_size:
        raise ValueError("Training world_size must divide the registered global batch 8")
    expected_hyperparameters = {
        "epochs": 1.0,
        "learning_rate": 1e-5,
        "global_batch_size": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8 // world_size,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": list(LORA_TARGET_MODULES),
        "precision": "bf16",
        "gradient_checkpointing": True,
        "max_length": MAX_LENGTH,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "seed": TRAINING_SEED,
        "data_seed": TRAINING_SEED,
        "loss": "assistant_tokens_only_prefix_difference",
        "report_to": "none",
    }
    if hyperparameters != expected_hyperparameters:
        raise ValueError(
            "Training hyperparameters differ from the registered Stage-7 contract"
        )
    if train_manifest.get("training_seed") != TRAINING_SEED:
        raise ValueError(f"Training manifest training_seed must be {TRAINING_SEED}")
    fingerprint_payload = {
        "contract": "when2tool-stage7-shared-training-protocol-v1",
        "sft_jsonl_sha256": hash_fields["sft_jsonl_sha256"],
        "sft_manifest_sha256": hash_fields["sft_manifest_sha256"],
        "primary_mask_sha256": hash_fields["neuron_mask_sha256"],
        "runtime_provenance_sha256": hash_fields["runtime_provenance_sha256"],
        "project_git_commit": project_git_commit,
        "base_model_slug": train_manifest["base_model"],
        "config_sha256": hash_fields["config_sha256"],
        "full_menu_sha256": hash_fields["full_menu_sha256"],
        "hyperparameters": hyperparameters,
        "world_size": world_size,
        "condition_specific_row_selection": "excluded_by_design",
    }
    if canonical_json_sha256(fingerprint_payload) != hash_fields["protocol_fingerprint"]:
        raise ValueError("Training protocol_fingerprint does not recompute")
    training_protocol = {
        **fingerprint_payload,
        "sft_input_bundle_sha256": hash_fields["sft_input_bundle_sha256"],
        "sft_n_records": sft_jsonl.get("n_records"),
        "neuron_mask_schema_version": neuron_mask.get("schema_version"),
        "dependency_versions": dependency_versions,
        "protocol_fingerprint": hash_fields["protocol_fingerprint"],
    }

    files = _adapter_files(root)
    bundle_sha256 = canonical_json_sha256(list(files))
    training_control = {
        "control_id": train_manifest["control_id"],
        "mode": train_manifest.get("mode"),
        "random_mask_seed": train_manifest.get("random_mask_seed"),
        "training_seed": train_manifest.get("training_seed"),
        "protocol_fingerprint": train_manifest.get("protocol_fingerprint"),
        "train_manifest_sha256": sha256_file(root / "train_manifest.json"),
        "row_selection": train_manifest.get("row_selection"),
    }
    return AdapterIdentity(
        condition_id=condition_id,
        kind="peft_adapter",
        bundle_sha256=bundle_sha256,
        files=files,
        training_control=training_control,
        training_protocol=training_protocol,
    )


def _default_peft_loader(
    base_model: torch.nn.Module, adapter_dir: Path, condition_id: str
) -> torch.nn.Module:
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError(
            "Adapter evaluation requires the optional peft package; no adapter "
            "fallback is permitted"
        ) from error
    return PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        adapter_name=condition_id,
        is_trainable=False,
        local_files_only=True,
    )


def load_adapter_into_agent(
    agent: HFCausalAgent,
    *,
    condition_id: str,
    adapter_dir: Path | str,
    loader: Callable[[torch.nn.Module, Path, str], torch.nn.Module] | None = None,
) -> HFCausalAgent:
    """Attach one local inference-only adapter without importing PEFT in tests."""

    condition_id = validate_condition_id(condition_id)
    if condition_id == BASE_CONDITION_ID:
        raise ValueError("Base condition must not call load_adapter_into_agent")
    root = Path(adapter_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    wrapped = (loader or _default_peft_loader)(agent.model, root, condition_id)
    if not isinstance(wrapped, torch.nn.Module):
        raise TypeError("PEFT loader must return a torch.nn.Module")
    wrapped.eval()
    trainable = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(
            "Inference adapter exposes trainable parameters: " + ", ".join(trainable[:20])
        )
    try:
        device = next(wrapped.parameters()).device
    except StopIteration as error:
        raise TypeError("Adapter-wrapped model has no parameters") from error
    if device != agent.device:
        raise ValueError(
            f"Adapter moved model from {agent.device} to {device}; device fallback is forbidden"
        )
    agent.model = wrapped
    return agent


def load_frozen_labels(
    path: Path | str, *, model_slug: str
) -> tuple[list[dict[str, Any]], str]:
    source = Path(path).resolve()
    artifact = _read_json_object(source, "label artifact")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TypeError("Label artifact rows must be a non-empty list")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": model_slug,
        "split": "test",
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        # Freeze one common label panel for both scoped and full evaluation.
        "tool_scope": "full",
        "n": len(rows),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"Frozen label artifact {key!r}={artifact.get(key)!r}; expected {value!r}"
            )
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or len(ids) != len(set(ids)):
        raise ValueError("Frozen label rows contain malformed or duplicate IDs")
    return rows, sha256_file(source)


def compute_adapter_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Action metrics compatible with causal evaluation plus training comparisons."""

    base = compute_ablation_metrics(rows, masked_class=None)
    needed = 0
    under_call = 0
    wrong_category = 0
    mixed = 0
    tool_need_tp = 0
    tool_need_fp = 0
    tool_need_fn = 0
    for index, row in enumerate(rows):
        gold = row.get("gold_action")
        pred = row.get("pred_action")
        gold_needs_tool = gold != "NONE"
        predicts_tool = pred != "NONE"
        tool_need_tp += int(gold_needs_tool and predicts_tool)
        tool_need_fp += int(not gold_needs_tool and predicts_tool)
        tool_need_fn += int(gold_needs_tool and not predicts_tool)
        if gold != "NONE":
            needed += 1
            under_call += int(pred == "NONE")
            wrong_category += int(pred in {"A", "B", "C"} and pred != gold)
        value = row.get("mixed_category_calls")
        if type(value) is not bool:
            raise TypeError(f"Row {index} mixed_category_calls must be bool")
        mixed += int(value)
    if needed == 0:
        raise ValueError("Adapter evaluation has no tool-needed examples")
    tool_need_denominator = 2 * tool_need_tp + tool_need_fp + tool_need_fn
    tool_need_f1 = (
        float(2 * tool_need_tp / tool_need_denominator)
        if tool_need_denominator
        else 0.0
    )
    metrics = {
        "N": base["N"],
        # Upstream When2Tool calls final-answer correctness Accuracy.  FinalAcc
        # is retained as the action-analysis spelling of the same observable.
        "Accuracy": base["FinalAcc"],
        "FinalAcc": base["FinalAcc"],
        "TotalTC": base["TotalTC"],
        "AvgTC": base["AvgTC"],
        # The registered single-hop tool-call rate uses calls/tasks.  It is
        # numerically identical to AvgTC and is kept under both names so the
        # When2Tool and action-analysis tables can be audited directly.
        "TCR": base["AvgTC"],
        "ToolNeed_F1": tool_need_f1,
        "ActionAcc": base["ActionAcc"],
        "MacroF1_action": base["MacroF1_action"],
        **{f"Recall_{action}": base[f"Recall_{action}"] for action in ACTIONS},
        "Recall4_macro": float(
            np.mean([base[f"Recall_{action}"] for action in ACTIONS])
        ),
        "OverCall": base["OverCall"],
        "UnderCall": float(under_call / needed),
        "WrongCat": float(wrong_category / needed),
        "Invalid": base["InvalidRate"],
        "Mixed": float(mixed / len(rows)),
    }
    return metrics


def summarize_adapter_metrics(
    per_seed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not per_seed_rows:
        raise ValueError("No per-seed adapter metrics")
    seeds = [row.get("generation_seed") for row in per_seed_rows]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("Every metric row requires integer generation_seed")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate generation seeds in adapter summary")
    metadata_keys = {"condition_id", "tool_scope", "generation_seed"}
    metric_names = sorted(
        key
        for key, value in per_seed_rows[0].items()
        if key not in metadata_keys
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )
    aggregate: dict[str, Any] = {"n_generation_seeds": len(per_seed_rows)}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in per_seed_rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite metric values for {name}")
        aggregate[f"{name}_mean"] = float(values.mean())
        aggregate[f"{name}_population_sd"] = float(values.std(ddof=0))
    return aggregate


def atomic_write_metric_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metric CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_adapter_summary(
    output_dir: Path | str,
    *,
    metadata: dict[str, Any],
    per_seed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    aggregate = summarize_adapter_metrics(per_seed_rows)
    summary = {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "metadata": metadata,
        "metric_contract": {
            "Accuracy": "upstream final-answer correctness",
            "FinalAcc": "alias of Accuracy for action-analysis tables",
            "TCR": (
                "registered single-hop tool-call rate TotalTC/N; numerically "
                "identical to AvgTC"
            ),
            "ToolNeed_F1": (
                "binary F1 where gold tool-needed iff gold_action!=NONE and "
                "predicted tool-needed iff pred_action!=NONE"
            ),
            "tc_reduction_and_accuracy_loss": (
                "computed only in the horizontal comparison against the "
                "same-scope transformers-hf base_model"
            ),
            "causal_attribution_backend": "transformers-hf",
            "vllm_probe_prefill": (
                "context-only comparison; backend differs and therefore cannot "
                "support direct causal attribution"
            ),
        },
        "per_seed_metrics": list(per_seed_rows),
        "aggregate": aggregate,
    }
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_metric_csv(destination / "per_seed_metrics.csv", per_seed_rows)
    atomic_write_metric_csv(
        destination / "summary.csv",
        [
            {
                "condition_id": metadata["condition_id"],
                "tool_scope": metadata["tool_scope"],
                **aggregate,
            }
        ],
    )
    atomic_write_json(destination / "summary.json", summary, overwrite=True)
    return summary


def load_completed_evaluation_cell(
    path: Path | str,
    *,
    expected_condition_id: str,
    expected_tool_scope: str,
) -> CompletedEvaluationCell:
    """Cross-check one manifest, summary, and every declared seed checkpoint."""

    condition_id = validate_condition_id(expected_condition_id)
    if expected_tool_scope not in {"scoped", "full"}:
        raise ValueError("expected_tool_scope must be scoped or full")
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = _read_json_object(root / "evaluation_manifest.json", "evaluation manifest")
    summary = _read_json_object(root / "summary.json", "evaluation summary")
    if manifest.get("schema_version") != ADAPTER_EVAL_SCHEMA_VERSION:
        raise ValueError(f"Unexpected manifest schema in {root}")
    if summary.get("schema_version") != ADAPTER_EVAL_SCHEMA_VERSION:
        raise ValueError(f"Unexpected summary schema in {root}")
    if manifest.get("manifest_type") != "trained-hf-evaluation":
        raise ValueError(f"Unexpected manifest type in {root}")
    if manifest.get("condition_id") != condition_id:
        raise ValueError(f"Manifest condition ID differs in {root}")
    if manifest.get("tool_scope") != expected_tool_scope:
        raise ValueError(f"Manifest tool scope differs in {root}")
    seeds = manifest.get("generation_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
        or not set(seeds) <= {0, 1, 2}
    ):
        raise ValueError(f"Invalid generation seed panel in {root}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"Manifest config is not an object in {root}")
    if config.get("backend") != "transformers-hf":
        raise ValueError(f"Non-HF backend in trained evaluation cell {root}")
    adapter = config.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("condition_id") != condition_id:
        raise ValueError(f"Adapter identity differs from condition in {root}")
    if config.get("tool_scope") != expected_tool_scope:
        raise ValueError(f"Config scope differs in {root}")

    metadata = summary.get("metadata")
    per_seed = summary.get("per_seed_metrics")
    aggregate = summary.get("aggregate")
    if not isinstance(metadata, dict) or not isinstance(per_seed, list) or not isinstance(
        aggregate, dict
    ):
        raise TypeError(f"Malformed summary sections in {root}")
    expected_summary_metadata = {
        "condition_id": condition_id,
        "tool_scope": expected_tool_scope,
        "backend": "transformers-hf",
        "adapter_bundle_sha256": adapter.get("bundle_sha256"),
        "data_sha256": config.get("data_sha256"),
        "labels_sha256": config.get("labels_sha256"),
        "generation_seeds": seeds,
        "vllm_probe_prefill_is_context_only": True,
    }
    if metadata != expected_summary_metadata:
        raise ValueError(f"Summary metadata differs from manifest in {root}")
    by_seed: dict[int, dict[str, Any]] = {}
    for row in per_seed:
        if not isinstance(row, dict):
            raise TypeError(f"Per-seed metric row is not an object in {root}")
        seed = row.get("generation_seed")
        if seed in by_seed:
            raise ValueError(f"Duplicate summary seed {seed} in {root}")
        if row.get("condition_id") != condition_id or row.get(
            "tool_scope"
        ) != expected_tool_scope:
            raise ValueError(f"Per-seed metric metadata differs in {root}")
        by_seed[seed] = row
    if set(by_seed) != set(seeds):
        raise ValueError(f"Summary seed panel differs from manifest in {root}")

    trajectory_dir = root / "trajectories"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(trajectory_dir)
    expected_names = {f"seed_{seed}.json" for seed in seeds}
    actual_names = {item.name for item in trajectory_dir.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            f"Trajectory file panel differs in {root}: "
            f"missing={sorted(expected_names-actual_names)}, "
            f"extra={sorted(actual_names-expected_names)}"
        )
    for seed in seeds:
        checkpoint = _read_json_object(
            trajectory_dir / f"seed_{seed}.json", "seed checkpoint"
        )
        if checkpoint.get("schema_version") != ADAPTER_EVAL_SCHEMA_VERSION:
            raise ValueError(f"Checkpoint schema differs for {root}/seed {seed}")
        if checkpoint.get("config") != config:
            raise ValueError(f"Checkpoint config differs for {root}/seed {seed}")
        expected_checkpoint_fields = {
            "condition_id": condition_id,
            "tool_scope": expected_tool_scope,
            "generation_seed": seed,
            "run_id": f"hf_{condition_id}_{expected_tool_scope}_seed_{seed}",
        }
        for key, value in expected_checkpoint_fields.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint {key} differs for {root}/seed {seed}")
        metrics = checkpoint.get("metrics")
        rows = checkpoint.get("rows")
        if not isinstance(metrics, dict) or not isinstance(rows, list) or not rows:
            raise TypeError(f"Checkpoint lacks metrics/rows for {root}/seed {seed}")
        if compute_adapter_metrics(rows) != metrics:
            raise ValueError(f"Checkpoint metrics do not recompute for {root}/seed {seed}")
        summary_metrics = {
            key: value
            for key, value in by_seed[seed].items()
            if key not in {"condition_id", "tool_scope", "generation_seed"}
        }
        if summary_metrics != metrics:
            raise ValueError(f"Summary/checkpoint metrics differ for {root}/seed {seed}")
    ordered_rows = tuple(by_seed[seed] for seed in seeds)
    if summarize_adapter_metrics(ordered_rows) != aggregate:
        raise ValueError(f"Aggregate metrics do not recompute in {root}")
    return CompletedEvaluationCell(
        condition_id=condition_id,
        tool_scope=expected_tool_scope,
        generation_seeds=tuple(seeds),
        config=config,
        per_seed_metrics=ordered_rows,
        aggregate=aggregate,
        path=root,
    )


def collect_trained_evaluation_grid(
    output_root: Path | str, *, require_complete: bool
) -> tuple[list[CompletedEvaluationCell], list[dict[str, str]]]:
    """Load the two-scope by six-condition grid and enforce shared bindings."""

    root = Path(output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    cells: list[CompletedEvaluationCell] = []
    missing: list[dict[str, str]] = []
    for scope in ("scoped", "full"):
        scope_dir = root / scope
        if scope_dir.is_dir():
            unknown = sorted(
                item.name
                for item in scope_dir.iterdir()
                if item.is_dir() and item.name not in CONDITION_IDS
            )
            if unknown:
                raise ValueError(f"Unknown trained-evaluation conditions in {scope}: {unknown}")
        for condition_id in CONDITION_IDS:
            path = scope_dir / condition_id
            if not path.is_dir():
                missing.append({"tool_scope": scope, "condition_id": condition_id})
                continue
            cells.append(
                load_completed_evaluation_cell(
                    path,
                    expected_condition_id=condition_id,
                    expected_tool_scope=scope,
                )
            )
    if require_complete and missing:
        raise FileNotFoundError(f"Incomplete 12-cell trained-evaluation grid: {missing}")
    if not cells:
        raise ValueError("No completed trained-evaluation cells found")

    seeds = {cell.generation_seeds for cell in cells}
    labels = {cell.config.get("labels_sha256") for cell in cells}
    backends = {cell.config.get("backend") for cell in cells}
    task_panels = {cell.config.get("task_ids_sha256") for cell in cells}
    config_hashes = {cell.config.get("config_sha256") for cell in cells}
    provenance = {
        (
            cell.config.get("runtime_provenance_sha256"),
            cell.config.get("project_git_commit"),
        )
        for cell in cells
    }
    for name, values in (
        ("generation seeds", seeds),
        ("frozen labels", labels),
        ("backend", backends),
        ("task ID panel", task_panels),
        ("experiment config", config_hashes),
        ("runtime provenance", provenance),
    ):
        if len(values) != 1 or None in values:
            raise ValueError(f"Trained-evaluation cells do not share {name}: {values}")
    if require_complete and seeds != {(0, 1, 2)}:
        raise ValueError(
            "Formal complete comparison requires generation seeds [0, 1, 2]"
        )
    if backends != {"transformers-hf"}:
        raise ValueError("Only transformers-hf cells may enter trained comparison")
    for scope in ("scoped", "full"):
        scope_data = {
            cell.config.get("data_sha256")
            for cell in cells
            if cell.tool_scope == scope
        }
        if len(scope_data) > 1 or None in scope_data:
            raise ValueError(f"{scope} conditions do not share one data artifact")
    shared_training_protocols: set[str] = set()
    for condition_id in CONDITION_IDS:
        condition_cells = [cell for cell in cells if cell.condition_id == condition_id]
        if not condition_cells:
            continue
        adapter_snapshots = [cell.config["adapter"] for cell in condition_cells]
        bundle_hashes = {item.get("bundle_sha256") for item in adapter_snapshots}
        control_hashes = {
            canonical_json_sha256(item.get("training_control"))
            for item in adapter_snapshots
        }
        if len(bundle_hashes) != 1 or len(control_hashes) != 1:
            raise ValueError(
                f"Scoped/full cells use different adapter artifact for {condition_id}"
            )
        if condition_id == BASE_CONDITION_ID:
            if any(
                item.get("bundle_sha256") is not None
                or item.get("training_control") is not None
                or item.get("training_protocol") is not None
                for item in adapter_snapshots
            ):
                raise ValueError("base_model cells must have null adapter training identity")
        else:
            if any(
                item.get("bundle_sha256") is None
                or not isinstance(item.get("training_control"), dict)
                or not isinstance(item.get("training_protocol"), dict)
                for item in adapter_snapshots
            ):
                raise ValueError(f"Adapter training identity missing for {condition_id}")
            shared_training_protocols.update(
                canonical_json_sha256(item["training_protocol"])
                for item in adapter_snapshots
            )
    if shared_training_protocols and len(shared_training_protocols) != 1:
        raise ValueError(
            "Non-base adapters do not share one frozen training protocol"
        )
    return cells, missing


def _build_comparison_summary_rows(
    cells: Sequence[CompletedEvaluationCell],
) -> list[dict[str, Any]]:
    """Add same-backend base deltas and a machine-readable scoped Pareto flag."""

    rows = [
        {
            "condition_id": cell.condition_id,
            "tool_scope": cell.tool_scope,
            **cell.aggregate,
        }
        for cell in cells
    ]
    by_scope = {
        scope: {
            row["condition_id"]: row
            for row in rows
            if row["tool_scope"] == scope
        }
        for scope in ("scoped", "full")
    }
    for row in rows:
        base = by_scope[row["tool_scope"]].get(BASE_CONDITION_ID)
        row["comparison_baseline_condition_id"] = (
            BASE_CONDITION_ID if base is not None else None
        )
        row["TC_reduction_vs_same_backend_base"] = (
            float(base["TotalTC_mean"] - row["TotalTC_mean"])
            if base is not None
            else None
        )
        row["Accuracy_loss_vs_same_backend_base"] = (
            float(base["Accuracy_mean"] - row["Accuracy_mean"])
            if base is not None
            else None
        )
        row["scoped_pareto_front"] = None

    scoped = by_scope["scoped"]
    if set(scoped) == set(CONDITION_IDS):
        for condition_id, row in scoped.items():
            tool_calls = float(row["TotalTC_mean"])
            accuracy = float(row["Accuracy_mean"])
            dominated = any(
                other_id != condition_id
                and float(other["TotalTC_mean"]) <= tool_calls
                and float(other["Accuracy_mean"]) >= accuracy
                and (
                    float(other["TotalTC_mean"]) < tool_calls
                    or float(other["Accuracy_mean"]) > accuracy
                )
                for other_id, other in scoped.items()
            )
            row["scoped_pareto_front"] = not dominated
    rows.sort(
        key=lambda row: (
            row["tool_scope"],
            CONDITION_IDS.index(row["condition_id"]),
        )
    )
    return rows


def _build_target_vs_controls(
    summary_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Create exploratory H3 target-minus-control effect-size rows."""

    comparison_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for scope in ("scoped", "full"):
        by_condition = {
            row["condition_id"]: row
            for row in summary_rows
            if row["tool_scope"] == scope
        }
        required = {
            BASE_CONDITION_ID,
            "target_neuron_lora",
            "dense_mlp_lora",
            *RANDOM_CONTROL_CONDITION_IDS,
        }
        missing_conditions = sorted(required - set(by_condition))
        if missing_conditions:
            missing.extend(
                {"tool_scope": scope, "condition_id": condition_id}
                for condition_id in missing_conditions
            )
            continue
        target = by_condition["target_neuron_lora"]
        base = by_condition[BASE_CONDITION_ID]
        dense = by_condition["dense_mlp_lora"]
        random_controls = [
            by_condition[condition_id]
            for condition_id in RANDOM_CONTROL_CONDITION_IDS
        ]
        for metric in H3_COMPARISON_METRICS:
            key = f"{metric}_mean"
            values = [target[key], base[key], dense[key]] + [
                row[key] for row in random_controls
            ]
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
                for value in values
            ):
                raise ValueError(f"H3 metric {key} is absent or non-finite in {scope}")
            target_value = float(target[key])
            base_value = float(base[key])
            dense_value = float(dense[key])
            random_mean = float(
                np.mean([float(row[key]) for row in random_controls])
            )
            comparison_rows.append(
                {
                    "tool_scope": scope,
                    "metric": metric,
                    "direction": (
                        "lower_is_better"
                        if metric
                        in {
                            "TotalTC",
                            "TCR",
                            "OverCall",
                            "UnderCall",
                            "WrongCat",
                            "Invalid",
                            "Mixed",
                        }
                        else "higher_is_better"
                    ),
                    "target_condition_id": "target_neuron_lora",
                    "target_mean": target_value,
                    "base_condition_id": BASE_CONDITION_ID,
                    "base_mean": base_value,
                    "dense_condition_id": "dense_mlp_lora",
                    "dense_mean": dense_value,
                    "random_condition_ids": ",".join(RANDOM_CONTROL_CONDITION_IDS),
                    "random_control_count": len(random_controls),
                    "random3_mean": random_mean,
                    "delta_target_minus_base": target_value - base_value,
                    "delta_target_minus_dense": target_value - dense_value,
                    "delta_target_minus_random3_mean": target_value - random_mean,
                    "analysis_status": "exploratory_effect_size_only",
                    "statistical_significance_tested": False,
                }
            )
    return comparison_rows, missing


def _plot_trained_comparison(
    summary_rows: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scoped = [row for row in summary_rows if row["tool_scope"] == "scoped"]
    full = [row for row in summary_rows if row["tool_scope"] == "full"]
    if len(scoped) == len(CONDITION_IDS):
        x = np.asarray([row["TotalTC_mean"] for row in scoped], dtype=float)
        y = np.asarray([row["Accuracy_mean"] for row in scoped], dtype=float)
        pareto = [bool(row["scoped_pareto_front"]) for row in scoped]
        figure, axis = plt.subplots(figsize=(9, 6))
        for index, row in enumerate(scoped):
            axis.errorbar(
                x[index],
                y[index],
                xerr=row["TotalTC_population_sd"],
                yerr=row["Accuracy_population_sd"],
                fmt="o" if pareto[index] else "x",
                capsize=3,
            )
            axis.annotate(row["condition_id"], (x[index], y[index]), xytext=(4, 4), textcoords="offset points")
        axis.set_xlabel("Total tool calls (lower is better)")
        axis.set_ylabel("Accuracy (higher is better)")
        axis.set_title("Scoped HF comparison: Accuracy vs TotalTC (Pareto points are circles)")
        figure.tight_layout()
        figure.savefig(output_dir / "scoped_accuracy_vs_totaltc.png", dpi=180)
        plt.close(figure)
    if len(full) == len(CONDITION_IDS):
        metrics = (
            "FinalAcc",
            "ActionAcc",
            "MacroF1_action",
            "Recall_B",
            "Recall_C",
            "OverCall",
        )
        x = np.arange(len(full))
        width = 0.12
        figure, axis = plt.subplots(figsize=(15, 6))
        for metric_index, metric in enumerate(metrics):
            offset = (metric_index - (len(metrics) - 1) / 2) * width
            axis.bar(
                x + offset,
                [row[f"{metric}_mean"] for row in full],
                width,
                yerr=[row[f"{metric}_population_sd"] for row in full],
                capsize=2,
                label=metric,
            )
        axis.set_xticks(x, [row["condition_id"] for row in full], rotation=25, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Rate")
        axis.set_title("Full-tools HF action metrics")
        axis.legend(ncol=3)
        figure.tight_layout()
        figure.savefig(output_dir / "full_action_metrics.png", dpi=180)
        plt.close(figure)


def publish_trained_comparison(
    output_root: Path | str,
    output_dir: Path | str,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Transactionally publish horizontal tables and HF-only comparison plots."""

    source = Path(output_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Comparison output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cells, missing = collect_trained_evaluation_grid(
        source, require_complete=require_complete
    )
    per_seed_rows = [dict(row) for cell in cells for row in cell.per_seed_metrics]
    summary_rows = _build_comparison_summary_rows(cells)
    target_vs_controls, missing_h3_cells = _build_target_vs_controls(summary_rows)
    per_seed_rows.sort(
        key=lambda row: (
            row["tool_scope"],
            CONDITION_IDS.index(row["condition_id"]),
            row["generation_seed"],
        )
    )
    first = cells[0]
    payload = {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "manifest_type": "trained-hf-horizontal-comparison",
        "panel_status": "complete" if not missing else "partial",
        "require_complete": require_complete,
        "completed_cells": len(cells),
        "expected_cells": 12,
        "missing_cells": missing,
        "generation_seeds": list(first.generation_seeds),
        "backend": "transformers-hf",
        "metric_contract": {
            "TCR": (
                "registered single-hop TotalTC/N; numerically identical to AvgTC"
            ),
            "TC_reduction_vs_same_backend_base": (
                "same-scope transformers-hf base TotalTC_mean minus condition "
                "TotalTC_mean; positive means fewer calls"
            ),
            "Accuracy_loss_vs_same_backend_base": (
                "same-scope transformers-hf base Accuracy_mean minus condition "
                "Accuracy_mean; positive means an accuracy loss"
            ),
            "scoped_pareto_front": (
                "non-dominated among all six scoped transformers-hf conditions "
                "for lower TotalTC_mean and higher Accuracy_mean; null for full "
                "scope or an incomplete scoped panel"
            ),
        },
        "labels_sha256": first.config["labels_sha256"],
        "data_sha256_by_scope": {
            scope: next(
                cell.config["data_sha256"]
                for cell in cells
                if cell.tool_scope == scope
            )
            for scope in {cell.tool_scope for cell in cells}
        },
        "vllm_probe_prefill_context": {
            "included_in_numeric_comparison": False,
            "reason": (
                "vLLM Probe&Prefill is a context-only baseline; backend mismatch "
                "precludes direct causal attribution and mixed aggregation"
            ),
            "external_context_table_entry": None,
        },
        "hypothesis_3": {
            "analysis_status": "exploratory_effect_size_only",
            "statistical_significance_tested": False,
            "interpretation": (
                "Target-minus-control deltas are descriptive effect sizes, not "
                "statistical-significance claims."
            ),
            "target_vs_controls_artifacts": [
                "target_vs_controls.csv",
                "target_vs_controls.json",
            ],
            "missing_required_cells": missing_h3_cells,
        },
        "condition_summaries": summary_rows,
    }
    target_payload = {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "manifest_type": "trained-hf-target-vs-controls",
        "analysis_status": "exploratory_effect_size_only",
        "statistical_significance_tested": False,
        "interpretation": (
            "Each delta is target_neuron_lora minus the named same-scope "
            "transformers-hf control mean. random3_mean equally averages the "
            "three registered random-mask adapter condition means. These are "
            "descriptive effect sizes and not significance tests."
        ),
        "random_control_condition_ids": list(RANDOM_CONTROL_CONDITION_IDS),
        "missing_required_cells": missing_h3_cells,
        "rows": target_vs_controls,
    }
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        atomic_write_metric_csv(stage / "comparison_per_seed.csv", per_seed_rows)
        atomic_write_metric_csv(stage / "comparison_summary.csv", summary_rows)
        atomic_write_json(stage / "comparison_summary.json", payload)
        if target_vs_controls:
            atomic_write_metric_csv(
                stage / "target_vs_controls.csv", target_vs_controls
            )
        atomic_write_json(stage / "target_vs_controls.json", target_payload)
        _plot_trained_comparison(summary_rows, stage)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return payload
