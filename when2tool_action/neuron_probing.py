"""Precise-Shield-style tool-action neuron selection and linear probes.

Neuron masks are selected exclusively from the train split.  Test activations
enter only :func:`fit_selected_feature_probes`, after a mask is already fixed.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .constants import ACTIONS, EXPECTED_SPLIT_SIZES, SCHEMA_VERSION, UPSTREAM_COMMIT
from .io_utils import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from .mlp_activations import (
    ACTIVATION_SCHEMA_VERSION,
    DOWN_NORMS_FILENAME,
    PINNED_TRANSFORMERS_VERSION,
    activation_filename,
    activation_manifest_filename,
    runtime_provenance_identity,
)


MASK_SCHEMA_VERSION = "when2tool-neuron-mask-v1"
ALLOWED_RHOS = (0.001, 0.003, 0.005)
ACTIVATION_VARIANTS = ("signed", "positive", "abs")
DEFAULT_CONTROL_SEED = 42
GROUP_FILES = (
    "tool_action_neurons.json",
    "top_neurons_by_layer.csv",
    "probe_results.json",
    "probe_model.pt",
    "neuron_layer_distribution.png",
    "category_overlap_jaccard.png",
    "saliency_heatmap_by_layer.png",
)


@dataclass(frozen=True)
class ActivationSplit:
    split: str
    tensor: torch.Tensor
    ids: tuple[int, ...]
    manifest: dict[str, Any]
    tensor_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ControlGroup:
    action: str
    derived_seed: int
    target_pool_size: int
    rest_pool_size: int
    nominal_n: int
    target_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    target_ids: tuple[int, ...]
    control_ids: tuple[int, ...]
    difficulty_allocation: dict[str, dict[str, int]]
    target_difficulty_counts: dict[str, int]
    control_difficulty_counts: dict[str, int]


def rho_tag(rho: float) -> str:
    return f"{rho:.3f}"


def group_name(rho: float, variant: str) -> str:
    if variant not in ACTIVATION_VARIANTS:
        raise ValueError(f"Unsupported activation variant {variant!r}")
    return f"rho{rho_tag(rho)}_{variant}"


def preflight_probe_outputs(
    output_dir: Path, rhos: Iterable[float], variants: Iterable[str]
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Probe output is not a directory: {output_dir}")
    collisions = [
        output_dir / group_name(rho, variant)
        for rho in rhos
        for variant in variants
        if (output_dir / group_name(rho, variant)).exists()
    ]
    if collisions:
        formatted = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(
            "Refusing to overwrite neuron-probing group directories:\n" + formatted
        )


def load_activation_split(
    activation_dir: Path,
    split: str,
    *,
    runtime_provenance: dict[str, Any],
) -> ActivationSplit:
    """Load one extraction artifact and verify every recorded invariant/hash."""

    if not activation_dir.is_dir():
        raise FileNotFoundError(activation_dir)
    tensor_path = activation_dir / activation_filename(split)
    manifest_path = activation_dir / activation_manifest_filename(split)
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path} must contain a JSON object")
    required = {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "split": split,
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
        "right_padding": True,
        "attention_implementation": "sdpa",
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "tensor_file": tensor_path.name,
        "dtype": "torch.float16",
        "down_proj_column_norms_file": DOWN_NORMS_FILENAME,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{manifest_path.name} {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    runtime_identity = runtime_provenance_identity(runtime_provenance)
    for key, expected in runtime_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{manifest_path.name} is bound to a different Stage-5 {key}"
            )
    if manifest.get("tensor_sha256") != sha256_file(tensor_path):
        raise ValueError(f"{tensor_path.name} SHA256 does not match its manifest")
    ids = manifest.get("ids")
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(task_id, int) for task_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"{manifest_path.name} has invalid IDs")
    if manifest.get("ids_sha256") != canonical_json_sha256(ids):
        raise ValueError(f"{manifest_path.name} ID hash mismatch")
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_path} does not contain a tensor")
    if tensor.dtype != torch.float16 or tensor.ndim != 3:
        raise ValueError(f"{tensor_path.name} must be float16 [N,L,I]")
    if list(tensor.shape) != manifest.get("shape"):
        raise ValueError(f"{tensor_path.name} shape differs from its manifest")
    if tensor.shape[0] != len(ids):
        raise ValueError(f"{tensor_path.name} row count differs from IDs")
    if tensor.shape[0] != EXPECTED_SPLIT_SIZES[split]:
        raise ValueError(
            f"{split} probing requires {EXPECTED_SPLIT_SIZES[split]} rows, "
            f"got {tensor.shape[0]}"
        )
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise TypeError(f"{manifest_path.name} model metadata must be an object")
    expected_tail = (
        model.get("num_hidden_layers"),
        model.get("intermediate_size"),
    )
    if tuple(tensor.shape[1:]) != expected_tail:
        raise ValueError(f"{tensor_path.name} shape differs from model metadata")
    return ActivationSplit(
        split=split,
        tensor=tensor.contiguous(),
        ids=tuple(ids),
        manifest=manifest,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
    )


def load_down_proj_norms(
    activation_dir: Path,
    train: ActivationSplit,
) -> tuple[torch.Tensor, Path]:
    """Load the norm tensor using only the train extraction manifest."""
    filename = train.manifest["down_proj_column_norms_file"]
    if Path(filename).name != filename or filename != DOWN_NORMS_FILENAME:
        raise ValueError("Invalid down_proj norm filename in activation manifest")
    path = activation_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != train.manifest["down_proj_column_norms_sha256"]:
        raise ValueError("down_proj column norm SHA256 mismatch")
    norms = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(norms, torch.Tensor) or norms.dtype != torch.float32:
        raise TypeError("down_proj column norms must be a float32 tensor")
    if list(norms.shape) != train.manifest["down_proj_column_norms_shape"]:
        raise ValueError("down_proj column norm shape mismatch")
    if tuple(norms.shape) != tuple(train.tensor.shape[1:]):
        raise ValueError("down_proj column norms do not match activations")
    if not torch.all(torch.isfinite(norms)) or torch.any(norms <= 0):
        raise ValueError("down_proj column norms must be finite and positive")
    return norms.contiguous(), path


def validate_test_activation_compatibility(
    train: ActivationSplit, test: ActivationSplit
) -> None:
    """Validate evaluation inputs after all train-only masks are frozen."""

    if train.tensor.shape[1:] != test.tensor.shape[1:]:
        raise ValueError("Train/test activation dimensions differ")
    if train.manifest["model"] != test.manifest["model"]:
        raise ValueError("Train/test model metadata differ")
    fields = (
        "down_proj_column_norms_file",
        "down_proj_column_norms_shape",
        "down_proj_column_norms_dtype",
        "down_proj_column_norms_sha256",
        "config_sha256",
        "model_config_sha256",
        "runtime_provenance_sha256",
        "project_git_commit",
    )
    for field in fields:
        if train.manifest.get(field) != test.manifest.get(field):
            raise ValueError(f"Train/test activation manifests disagree on {field}")


def load_ordered_label_rows(
    label_path: Path,
    *,
    split: str,
    ids: tuple[int, ...],
    expected_sha256: str,
    model_slug: str,
) -> list[dict[str, Any]]:
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    if sha256_file(label_path) != expected_sha256:
        raise ValueError(
            f"{split} label artifact does not match the activation manifest hash"
        )
    artifact = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not isinstance(artifact.get("rows"), list):
        raise TypeError(f"{label_path} is not a label artifact")
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "split": split,
        "model": model_slug,
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
    }
    for key, expected in expected_top.items():
        if artifact.get(key) != expected:
            raise ValueError(f"{label_path.name} has invalid {key}")
    rows = artifact["rows"]
    if artifact.get("n") != len(rows):
        raise ValueError(f"{label_path.name} n does not match rows")
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            raise TypeError(f"Malformed row in {label_path}")
        if row["id"] in by_id:
            raise ValueError(f"Duplicate label ID {row['id']}")
        action = row.get("gold_action")
        if action not in ACTIONS:
            raise ValueError(f"Label {row['id']} has invalid gold_action")
        if row.get("tool_necessary") != int(action != "NONE"):
            raise ValueError(f"Label {row['id']} action/necessity mismatch")
        if row.get("split") != split:
            raise ValueError(f"Label {row['id']} split mismatch")
        if not isinstance(row.get("difficulty"), str) or not row["difficulty"]:
            raise ValueError(f"Label {row['id']} has invalid difficulty")
        by_id[row["id"]] = row
    if set(by_id) != set(ids) or len(by_id) != len(ids):
        raise ValueError(f"{split} activation/label ID sets differ")
    return [by_id[task_id] for task_id in ids]


def build_stratified_controls(
    label_rows: list[dict[str, Any]], *, seed: int
) -> dict[str, ControlGroup]:
    """Build maximum feasible, exactly difficulty-matched one-vs-rest pairs.

    Both sides are sampled without replacement.  For difficulty ``d`` the
    deterministic maximum feasible quota is ``min(|D_c,d|, |D_rest,d|)``.
    Consequently the actual total can be smaller than the global nominal
    ``min(|D_c|, |D_rest|)`` when difficulty distributions are incompatible.
    """

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("control seed must be a non-negative integer")
    ids = [row.get("id") for row in label_rows]
    if not ids or not all(isinstance(task_id, int) for task_id in ids):
        raise TypeError("Every label row must have an integer ID")
    if len(ids) != len(set(ids)):
        raise ValueError("Label IDs must be unique")
    for row in label_rows:
        if row.get("gold_action") not in ACTIONS:
            raise ValueError(f"Invalid gold_action for label {row['id']}")
        if not isinstance(row.get("difficulty"), str) or not row["difficulty"]:
            raise ValueError(f"Invalid difficulty for label {row['id']}")

    groups: dict[str, ControlGroup] = {}
    for action_index, action in enumerate(ACTIONS):
        target_pool_indices = tuple(
            index
            for index, row in enumerate(label_rows)
            if row["gold_action"] == action
        )
        rest_pool_indices = tuple(
            index
            for index, row in enumerate(label_rows)
            if row["gold_action"] != action
        )
        if not target_pool_indices or not rest_pool_indices:
            raise ValueError(f"Train labels contain no {action} examples")
        derived_seed = seed + action_index * 100_003
        rng = np.random.default_rng(derived_seed)
        selected_target: list[int] = []
        selected_control: list[int] = []
        allocation: dict[str, dict[str, int]] = {}
        difficulties = sorted(
            {
                row["difficulty"]
                for row in label_rows
            }
        )
        for difficulty in difficulties:
            target_pool = np.asarray(
                [
                    index
                    for index in target_pool_indices
                    if label_rows[index]["difficulty"] == difficulty
                ],
                dtype=np.int64,
            )
            rest_pool = np.asarray(
                [
                    index
                    for index in rest_pool_indices
                    if label_rows[index]["difficulty"] == difficulty
                ],
                dtype=np.int64,
            )
            quota = min(len(target_pool), len(rest_pool))
            allocation[difficulty] = {
                "target_available": len(target_pool),
                "control_available": len(rest_pool),
                "used_each_side": quota,
            }
            if quota:
                chosen_target = rng.choice(target_pool, size=quota, replace=False)
                chosen_control = rng.choice(rest_pool, size=quota, replace=False)
                selected_target.extend(
                    int(index) for index in chosen_target.tolist()
                )
                selected_control.extend(
                    int(index) for index in chosen_control.tolist()
                )
        target_indices = tuple(sorted(selected_target))
        control_indices = tuple(sorted(selected_control))
        if not target_indices:
            raise ValueError(
                f"{action} has no feasible difficulty-matched target/control pairs"
            )
        target_difficulty = Counter(
            label_rows[index]["difficulty"] for index in target_indices
        )
        control_difficulty = Counter(
            label_rows[index]["difficulty"] for index in control_indices
        )
        if control_difficulty != target_difficulty:
            raise AssertionError(f"{action} control difficulty matching failed")
        if any(label_rows[index]["gold_action"] == action for index in control_indices):
            raise AssertionError(f"{action} control contains target examples")
        groups[action] = ControlGroup(
            action=action,
            derived_seed=derived_seed,
            target_pool_size=len(target_pool_indices),
            rest_pool_size=len(rest_pool_indices),
            nominal_n=min(len(target_pool_indices), len(rest_pool_indices)),
            target_indices=target_indices,
            control_indices=control_indices,
            target_ids=tuple(label_rows[index]["id"] for index in target_indices),
            control_ids=tuple(label_rows[index]["id"] for index in control_indices),
            difficulty_allocation=allocation,
            target_difficulty_counts=dict(sorted(target_difficulty.items())),
            control_difficulty_counts=dict(sorted(control_difficulty.items())),
        )
    return groups


def activation_variant_mean(
    activations: torch.Tensor,
    indices: tuple[int, ...],
    *,
    variant: str,
    chunk_size: int,
) -> torch.Tensor:
    """Compute a float32 variant mean without materializing the full subset."""

    if variant not in ACTIVATION_VARIANTS:
        raise ValueError(f"Unsupported activation variant {variant!r}")
    if activations.ndim != 3 or not indices:
        raise ValueError("activations must be [N,L,I] and indices must be non-empty")
    if chunk_size < 1:
        raise ValueError("mean chunk_size must be positive")
    if min(indices) < 0 or max(indices) >= activations.shape[0]:
        raise IndexError("Activation subset index is out of range")
    total = torch.zeros(activations.shape[1:], dtype=torch.float32)
    for start in range(0, len(indices), chunk_size):
        batch_indices = list(indices[start : start + chunk_size])
        values = activations[batch_indices].to(dtype=torch.float32)
        if variant == "positive":
            values = values.clamp_min(0)
        elif variant == "abs":
            values = values.abs()
        total.add_(values.sum(dim=0))
    mean = total.div_(len(indices))
    if not torch.all(torch.isfinite(mean)):
        raise ValueError("Activation mean contains non-finite values")
    return mean


def compute_saliency_bundle(
    train_activations: torch.Tensor,
    down_norms: torch.Tensor,
    controls: dict[str, ControlGroup],
    *,
    variant: str,
    chunk_size: int = 64,
    epsilon: float = 1e-12,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute train-only class/control means, importance, and saliency."""

    if train_activations.ndim != 3 or train_activations.dtype != torch.float16:
        raise ValueError("Train activations must be a float16 [N,L,I] tensor")
    if down_norms.dtype != torch.float32 or tuple(down_norms.shape) != tuple(
        train_activations.shape[1:]
    ):
        raise ValueError("down_proj norms must be float32 [L,I]")
    if set(controls) != set(ACTIONS):
        raise ValueError("Control plan must cover NONE/A/B/C")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for action in ACTIONS:
        group = controls[action]
        class_mean = activation_variant_mean(
            train_activations,
            group.target_indices,
            variant=variant,
            chunk_size=chunk_size,
        )
        control_mean = activation_variant_mean(
            train_activations,
            group.control_indices,
            variant=variant,
            chunk_size=chunk_size,
        )
        # ||mean_i * W_down[:, i]||_2 = abs(mean_i) * ||W_down[:, i]||_2.
        class_importance = class_mean.abs() * down_norms
        control_importance = control_mean.abs() * down_norms
        class_totals = class_importance.sum(dim=1, keepdim=True)
        control_totals = control_importance.sum(dim=1, keepdim=True)
        if torch.any(class_totals <= 0) or torch.any(control_totals <= 0):
            raise ValueError(f"{action} has an all-zero saliency layer")
        result[action] = {
            "mean_class": class_mean,
            "mean_control": control_mean,
            "importance_class": class_importance,
            "importance_control": control_importance,
            "saliency_class": class_importance / (class_totals + epsilon),
            "saliency_control": control_importance / (control_totals + epsilon),
            "direction": torch.sign(class_mean - control_mean).to(torch.int8),
        }
    return result


def _stable_topk_indices(saliency: torch.Tensor, k: int) -> list[list[int]]:
    if saliency.ndim != 2 or k < 0 or k > saliency.shape[1]:
        raise ValueError("Invalid per-layer top-k request")
    if k == 0:
        return [[] for _ in range(saliency.shape[0])]
    # Stable sorting makes lower neuron indices win exact-score ties.
    ranked = torch.argsort(saliency, dim=1, descending=True, stable=True)[:, :k]
    return [[int(value) for value in row.tolist()] for row in ranked]


def _jaccard_matrix(class_sets: dict[str, set[tuple[int, int]]]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left in ACTIONS:
        row: list[float] = []
        for right in ACTIONS:
            union = class_sets[left] | class_sets[right]
            value = 1.0 if not union else len(class_sets[left] & class_sets[right]) / len(union)
            row.append(float(value))
        matrix.append(row)
    return matrix


def build_mask_artifact(
    scores: dict[str, dict[str, torch.Tensor]],
    down_norms: torch.Tensor,
    controls: dict[str, ControlGroup],
    *,
    rho: float,
    variant: str,
    model_metadata: dict[str, Any],
    config_metadata: dict[str, Any],
    selection_hashes: dict[str, str],
) -> dict[str, Any]:
    """Apply per-layer top-k then exact set difference, with no refill."""

    if not 0 < rho < 1:
        raise ValueError("rho must be in (0,1)")
    if variant not in ACTIVATION_VARIANTS:
        raise ValueError(f"Unsupported activation variant {variant!r}")
    if set(scores) != set(ACTIONS) or set(controls) != set(ACTIONS):
        raise ValueError("Scores and controls must cover NONE/A/B/C")
    n_layers, intermediate_size = down_norms.shape
    k = math.floor(rho * intermediate_size)
    if k < 1:
        raise ValueError(
            f"rho={rho} selects k=0 for intermediate_size={intermediate_size}"
        )
    classes: dict[str, Any] = {}
    class_sets: dict[str, set[tuple[int, int]]] = {}
    layer_counts: dict[str, dict[str, int]] = {}
    for action in ACTIONS:
        score = scores[action]
        for field in (
            "mean_class",
            "mean_control",
            "saliency_class",
            "saliency_control",
            "direction",
        ):
            if tuple(score[field].shape) != (n_layers, intermediate_size):
                raise ValueError(f"{action} score field {field} has the wrong shape")
        class_top = _stable_topk_indices(score["saliency_class"], k)
        control_top = _stable_topk_indices(score["saliency_control"], k)
        layers: dict[str, list[dict[str, Any]]] = {}
        selected_set: set[tuple[int, int]] = set()
        counts: dict[str, int] = {}
        for layer_index in range(n_layers):
            # Deliberately do not refill after removing shared control neurons.
            selected = sorted(set(class_top[layer_index]) - set(control_top[layer_index]))
            entries: list[dict[str, Any]] = []
            for neuron_index in selected:
                entries.append(
                    {
                        "neuron_idx": neuron_index,
                        "direction": int(score["direction"][layer_index, neuron_index]),
                        "saliency_class": float(
                            score["saliency_class"][layer_index, neuron_index]
                        ),
                        "saliency_control": float(
                            score["saliency_control"][layer_index, neuron_index]
                        ),
                        "mean_act_class": float(
                            score["mean_class"][layer_index, neuron_index]
                        ),
                        "mean_act_control": float(
                            score["mean_control"][layer_index, neuron_index]
                        ),
                        "down_norm": float(down_norms[layer_index, neuron_index]),
                    }
                )
                selected_set.add((layer_index, neuron_index))
            layers[str(layer_index)] = entries
            counts[str(layer_index)] = len(entries)
        classes[action] = {
            "target_examples": len(controls[action].target_indices),
            "control_examples": len(controls[action].control_indices),
            "topk_per_layer_before_set_difference": k,
            "total_neurons": len(selected_set),
            "layer_counts": counts,
            "layers": layers,
        }
        class_sets[action] = selected_set
        layer_counts[action] = counts

    union = set().union(*class_sets.values())
    control_metadata = {
        "strategy": (
            "one-vs-rest without replacement; per-difficulty maximum feasible "
            "quota min(target_available, control_available)"
        ),
        "base_seed": config_metadata["control_seed"],
        "classes": {
            action: {
                "derived_seed": controls[action].derived_seed,
                "target_pool_size": controls[action].target_pool_size,
                "rest_pool_size": controls[action].rest_pool_size,
                "nominal_n_min_target_rest": controls[action].nominal_n,
                "actual_n_each_side": len(controls[action].target_indices),
                "difficulty_allocation": controls[action].difficulty_allocation,
                "target_ids": list(controls[action].target_ids),
                "target_ids_sha256": canonical_json_sha256(
                    list(controls[action].target_ids)
                ),
                "control_ids": list(controls[action].control_ids),
                "control_ids_sha256": canonical_json_sha256(
                    list(controls[action].control_ids)
                ),
                "target_difficulty_counts": controls[action].target_difficulty_counts,
                "control_difficulty_counts": controls[action].control_difficulty_counts,
            }
            for action in ACTIONS
        },
    }
    return {
        "schema_version": MASK_SCHEMA_VERSION,
        "rho": rho,
        "activation_variant": variant,
        "runtime_provenance_sha256": selection_hashes[
            "runtime_provenance_sha256"
        ],
        "project_git_commit": config_metadata["project_git_commit"],
        "model": model_metadata,
        "config": {
            **config_metadata,
            "selection_split": "train",
            "test_used_for_selection": False,
            "importance": "abs(variant_mean_activation) * down_proj_column_l2_norm",
            "saliency": "importance / (layer_importance_sum + 1e-12)",
            "topk_rule": "floor(rho * intermediate_size) independently per layer",
            "set_difference": "class_topk minus control_topk; no refill",
            "topk_per_layer": k,
        },
        "hashes": selection_hashes,
        "control": control_metadata,
        "class_order": list(ACTIONS),
        "layer_counts": layer_counts,
        "overlap_jaccard": {
            "class_order": list(ACTIONS),
            "matrix": _jaccard_matrix(class_sets),
        },
        "total_neurons": len(union),
        "total_class_assignments": sum(len(values) for values in class_sets.values()),
        "union_features_sha256": canonical_json_sha256(
            [[layer, neuron] for layer, neuron in sorted(union)]
        ),
        "classes": classes,
    }


def selected_union(mask: dict[str, Any]) -> list[tuple[int, int]]:
    if mask.get("schema_version") != MASK_SCHEMA_VERSION:
        raise ValueError("Unsupported neuron mask schema")
    classes = mask.get("classes")
    if not isinstance(classes, dict) or set(classes) != set(ACTIONS):
        raise ValueError("Neuron mask classes must be exactly NONE/A/B/C")
    features: set[tuple[int, int]] = set()
    for action in ACTIONS:
        layers = classes[action].get("layers")
        if not isinstance(layers, dict):
            raise TypeError(f"Mask class {action} layers must be an object")
        for layer_text, entries in layers.items():
            layer_index = int(layer_text)
            if not isinstance(entries, list):
                raise TypeError(f"Mask layer {layer_text} must be a list")
            seen: set[int] = set()
            for entry in entries:
                neuron_index = entry.get("neuron_idx") if isinstance(entry, dict) else None
                if not isinstance(neuron_index, int) or neuron_index in seen:
                    raise ValueError(f"Invalid/duplicate neuron in {action} layer {layer_text}")
                seen.add(neuron_index)
                features.add((layer_index, neuron_index))
    return sorted(features)


def _metric_report(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_classes: int,
) -> dict[str, float]:
    labels = np.arange(n_classes)
    if set(np.unique(y_true)) != set(labels.tolist()):
        raise ValueError("Test labels must contain every registered class for AUROC")
    if n_classes == 2:
        auroc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        auroc = roc_auc_score(
            y_true,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
    counts = np.bincount(y_true, minlength=n_classes)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": float(auroc),
        "majority_baseline": float(counts.max() / counts.sum()),
    }


def fit_selected_feature_probes(
    train_activations: torch.Tensor,
    test_activations: torch.Tensor,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    mask: dict[str, Any],
    *,
    c: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on train only and evaluate fixed binary/four-class L2 probes on test."""

    if c <= 0:
        raise ValueError("Logistic-regression C must be positive")
    if train_activations.ndim != 3 or test_activations.ndim != 3:
        raise ValueError("Probe activations must be [N,L,I]")
    if train_activations.shape[1:] != test_activations.shape[1:]:
        raise ValueError("Train/test activation feature shapes differ")
    if len(train_rows) != train_activations.shape[0] or len(test_rows) != test_activations.shape[0]:
        raise ValueError("Activation/label row counts differ")
    features = selected_union(mask)
    if not features:
        raise ValueError("Cannot fit probes on an empty selected-neuron union")
    n_layers, intermediate_size = train_activations.shape[1:]
    if any(
        layer < 0
        or layer >= n_layers
        or neuron < 0
        or neuron >= intermediate_size
        for layer, neuron in features
    ):
        raise IndexError("Neuron mask feature is outside activation dimensions")
    flat_indices = torch.tensor(
        [layer * intermediate_size + neuron for layer, neuron in features],
        dtype=torch.long,
    )
    train_x = (
        train_activations.reshape(len(train_rows), -1)[:, flat_indices]
        .to(dtype=torch.float32)
        .numpy()
    )
    test_x = (
        test_activations.reshape(len(test_rows), -1)[:, flat_indices]
        .to(dtype=torch.float32)
        .numpy()
    )
    action_to_index = {action: index for index, action in enumerate(ACTIONS)}
    train_four = np.asarray(
        [action_to_index[row["gold_action"]] for row in train_rows], dtype=np.int64
    )
    test_four = np.asarray(
        [action_to_index[row["gold_action"]] for row in test_rows], dtype=np.int64
    )
    train_binary = (train_four != action_to_index["NONE"]).astype(np.int64)
    test_binary = (test_four != action_to_index["NONE"]).astype(np.int64)
    for name, labels, expected in (
        ("binary train", train_binary, {0, 1}),
        ("binary test", test_binary, {0, 1}),
        ("four-class train", train_four, set(range(len(ACTIONS)))),
        ("four-class test", test_four, set(range(len(ACTIONS)))),
    ):
        if set(np.unique(labels)) != expected:
            raise ValueError(f"{name} is missing registered classes")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)

    def fit_one(
        train_y: np.ndarray, test_y: np.ndarray, n_classes: int
    ) -> tuple[dict[str, float], LogisticRegression]:
        classifier = LogisticRegression(
            penalty="l2",
            C=c,
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
        classifier.fit(train_scaled, train_y)
        if not np.array_equal(classifier.classes_, np.arange(n_classes)):
            raise AssertionError("LogisticRegression class order changed")
        predictions = classifier.predict(test_scaled)
        probabilities = classifier.predict_proba(test_scaled)
        return (
            _metric_report(
                test_y, predictions, probabilities, n_classes=n_classes
            ),
            classifier,
        )

    binary_metrics, binary_model = fit_one(train_binary, test_binary, 2)
    four_metrics, four_model = fit_one(train_four, test_four, len(ACTIONS))
    feature_rows = [[layer, neuron] for layer, neuron in features]
    report = {
        "protocol": "mask fixed on train; StandardScaler/L2 logistic fit on train; test evaluation only",
        "C": c,
        "random_state": seed,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_union_features": len(features),
        "union_features_sha256": canonical_json_sha256(feature_rows),
        "binary_tool_needed": {
            "classes": ["NONE", "TOOL"],
            **binary_metrics,
        },
        "four_class_action": {
            "classes": list(ACTIONS),
            **four_metrics,
        },
    }
    model_payload = {
        "C": c,
        "random_state": seed,
        "feature_pairs": torch.tensor(feature_rows, dtype=torch.long),
        "scaler_mean": torch.from_numpy(scaler.mean_.copy()),
        "scaler_scale": torch.from_numpy(scaler.scale_.copy()),
        "binary": {
            "classes": ["NONE", "TOOL"],
            "coef": torch.from_numpy(binary_model.coef_.copy()),
            "intercept": torch.from_numpy(binary_model.intercept_.copy()),
        },
        "four_class": {
            "classes": list(ACTIONS),
            "coef": torch.from_numpy(four_model.coef_.copy()),
            "intercept": torch.from_numpy(four_model.intercept_.copy()),
        },
    }
    return report, model_payload


def _mask_table_rows(mask: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in ACTIONS:
        layers = mask["classes"][action]["layers"]
        for layer_text in sorted(layers, key=int):
            for entry in layers[layer_text]:
                rows.append(
                    {
                        "class": action,
                        "layer": int(layer_text),
                        **entry,
                    }
                )
    return rows


def _atomic_save_figure(fig: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".png.tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, format="png", dpi=200, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_plots(mask: dict[str, Any], output_dir: Path) -> None:
    n_layers = int(mask["model"]["num_hidden_layers"])
    layer_axis = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    for action in ACTIONS:
        counts = mask["classes"][action]["layer_counts"]
        ax.plot(
            layer_axis,
            [counts[str(layer)] for layer in layer_axis],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=action,
        )
    ax.set(
        xlabel="MLP layer",
        ylabel="Selected neurons after set difference",
        title=f"Tool-action neuron distribution ({mask['activation_variant']}, rho={mask['rho']})",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _atomic_save_figure(fig, output_dir / "neuron_layer_distribution.png")
    plt.close(fig)

    overlap = np.asarray(mask["overlap_jaccard"]["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(overlap, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(ACTIONS)), labels=ACTIONS)
    ax.set_yticks(range(len(ACTIONS)), labels=ACTIONS)
    ax.set_title("Selected-neuron Jaccard overlap")
    for row in range(len(ACTIONS)):
        for column in range(len(ACTIONS)):
            ax.text(column, row, f"{overlap[row, column]:.2f}", ha="center", va="center", color="white" if overlap[row, column] < 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Jaccard")
    fig.tight_layout()
    _atomic_save_figure(fig, output_dir / "category_overlap_jaccard.png")
    plt.close(fig)

    saliency = np.zeros((len(ACTIONS), n_layers), dtype=float)
    for action_index, action in enumerate(ACTIONS):
        for layer in range(n_layers):
            saliency[action_index, layer] = sum(
                entry["saliency_class"]
                for entry in mask["classes"][action]["layers"][str(layer)]
            )
    fig, ax = plt.subplots(figsize=(12, 3.8))
    image = ax.imshow(saliency, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(ACTIONS)), labels=ACTIONS)
    ax.set_xlabel("MLP layer")
    ax.set_title("Sum of selected class saliency by layer")
    fig.colorbar(image, ax=ax, label="Selected saliency sum")
    fig.tight_layout()
    _atomic_save_figure(fig, output_dir / "saliency_heatmap_by_layer.png")
    plt.close(fig)


def write_probe_group(
    output_dir: Path,
    mask: dict[str, Any],
    probe_report: dict[str, Any],
    probe_model: dict[str, Any],
) -> Path:
    """Build a complete group in a private directory, then publish atomically."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        atomic_write_json(temporary / "tool_action_neurons.json", mask)
        table = _mask_table_rows(mask)
        if not table:
            raise ValueError("Neuron mask contains no selected neurons")
        fieldnames = [
            "class",
            "layer",
            "neuron_idx",
            "direction",
            "saliency_class",
            "saliency_control",
            "mean_act_class",
            "mean_act_control",
            "down_norm",
        ]
        atomic_write_csv(
            temporary / "top_neurons_by_layer.csv", table, fieldnames
        )
        atomic_write_json(temporary / "probe_results.json", probe_report)
        atomic_torch_save(temporary / "probe_model.pt", probe_model)
        _write_plots(mask, temporary)
        produced = {path.name for path in temporary.iterdir() if path.is_file()}
        if produced != set(GROUP_FILES):
            raise AssertionError(
                f"Incomplete probe group; produced={sorted(produced)}"
            )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir


def _validate_requested_panel(
    rhos: tuple[float, ...], variants: tuple[str, ...]
) -> None:
    if len(rhos) != len(set(rhos)) or set(rhos) - set(ALLOWED_RHOS):
        raise ValueError(f"rhos must be a unique subset of {ALLOWED_RHOS}")
    if len(variants) != len(set(variants)) or set(variants) - set(ACTIVATION_VARIANTS):
        raise ValueError(
            f"variants must be a unique subset of {ACTIVATION_VARIANTS}"
        )
    if not rhos or not variants:
        raise ValueError("At least one rho and one activation variant are required")


def run_probing_suite(
    *,
    config: ExperimentConfig,
    activation_dir: Path,
    train_labels_path: Path,
    test_labels_path: Path,
    output_dir: Path,
    rhos: tuple[float, ...] = ALLOWED_RHOS,
    variants: tuple[str, ...] = ACTIVATION_VARIANTS,
    control_seed: int = DEFAULT_CONTROL_SEED,
    probe_c: float = 0.0001,
    mean_chunk_size: int = 64,
    runtime_provenance: dict[str, Any],
) -> list[Path]:
    """Run the frozen rho/variant panel and return independent group directories."""

    _validate_requested_panel(rhos, variants)
    preflight_probe_outputs(output_dir, rhos, variants)
    runtime_identity = runtime_provenance_identity(runtime_provenance)

    # Selection phase: no test tensor, test manifest, or test label is opened
    # until every requested mask has been fully materialized below.
    train = load_activation_split(
        activation_dir, "train", runtime_provenance=runtime_provenance
    )
    model_metadata = train.manifest["model"]
    expected_model = {
        "slug": config.model.slug,
        "architecture": config.model.architecture,
        "num_hidden_layers": config.model.num_hidden_layers,
        "hidden_size": config.model.hidden_size,
    }
    for key, expected in expected_model.items():
        if model_metadata.get(key) != expected:
            raise ValueError(f"Activation model metadata mismatch for {key}")
    norms, norms_path = load_down_proj_norms(activation_dir, train)
    train_rows = load_ordered_label_rows(
        train_labels_path,
        split="train",
        ids=train.ids,
        expected_sha256=train.manifest["labels_sha256"],
        model_slug=config.model.slug,
    )
    controls = build_stratified_controls(train_rows, seed=control_seed)
    selection_hashes = {
        "config_sha256": sha256_file(config.source),
        "train_activation_sha256": sha256_file(train.tensor_path),
        "train_activation_manifest_sha256": sha256_file(train.manifest_path),
        "train_labels_sha256": sha256_file(train_labels_path),
        "down_proj_column_norms_sha256": sha256_file(norms_path),
        "runtime_provenance_sha256": runtime_identity[
            "runtime_provenance_sha256"
        ],
    }
    config_metadata = {
        "control_seed": control_seed,
        "mean_chunk_size": mean_chunk_size,
        "probe_C": probe_c,
        "project_git_commit": runtime_identity["project_git_commit"],
    }
    masks: dict[tuple[str, float], dict[str, Any]] = {}
    for variant in variants:
        scores = compute_saliency_bundle(
            train.tensor,
            norms,
            controls,
            variant=variant,
            chunk_size=mean_chunk_size,
        )
        for rho in rhos:
            mask = build_mask_artifact(
                scores,
                norms,
                controls,
                rho=rho,
                variant=variant,
                model_metadata=model_metadata,
                config_metadata=config_metadata,
                selection_hashes=selection_hashes,
            )
            empty_classes = [
                action
                for action in ACTIONS
                if mask["classes"][action]["total_neurons"] == 0
            ]
            if empty_classes:
                raise ValueError(
                    f"rho={rho}/{variant} selected no neurons for {empty_classes}; no refill is permitted"
                )
            masks[(variant, rho)] = mask
    del scores

    # Evaluation phase starts only after all train-selected masks are frozen.
    test = load_activation_split(
        activation_dir, "test", runtime_provenance=runtime_provenance
    )
    validate_test_activation_compatibility(train, test)
    test_rows = load_ordered_label_rows(
        test_labels_path,
        split="test",
        ids=test.ids,
        expected_sha256=test.manifest["labels_sha256"],
        model_slug=config.model.slug,
    )
    evaluation_hashes = {
        "test_activation_sha256": sha256_file(test.tensor_path),
        "test_activation_manifest_sha256": sha256_file(test.manifest_path),
        "test_labels_sha256": sha256_file(test_labels_path),
        "runtime_provenance_sha256": runtime_identity[
            "runtime_provenance_sha256"
        ],
    }
    written: list[Path] = []
    for variant in variants:
        for rho in rhos:
            mask = masks[(variant, rho)]
            report, model_payload = fit_selected_feature_probes(
                train.tensor,
                test.tensor,
                train_rows,
                test_rows,
                mask,
                c=probe_c,
                seed=control_seed,
            )
            report.update(
                {
                    "schema_version": "when2tool-neuron-probe-results-v1",
                    "rho": rho,
                    "activation_variant": variant,
                    "selection_hashes": selection_hashes,
                    "evaluation_hashes": evaluation_hashes,
                    "mask_canonical_json_sha256": canonical_json_sha256(mask),
                    **runtime_identity,
                }
            )
            model_payload.update(runtime_identity)
            target = output_dir / group_name(rho, variant)
            written.append(write_probe_group(target, mask, report, model_payload))
    return written
