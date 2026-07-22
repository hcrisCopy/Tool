"""Mask construction, PyTorch hooks, metrics, and reports for causal tests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .constants import ACTIONS
from .io_utils import atomic_write_json, canonical_json_sha256


NEURON_MASK_SCHEMA_VERSION = "when2tool-neuron-mask-v1"
ABLATION_SCHEMA_VERSION = "when2tool-neuron-ablation-v1"
RANDOM_MASK_SEEDS = (0, 1, 2, 3, 4)
PREDICTIONS = (*ACTIONS, "INVALID")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ABLATION_METRIC_CONTRACT = {
    "UnderCall": "P(pred_action=NONE | gold_action in {A,B,C}); denominator ToolNeededN",
    "WrongCat": (
        "P(pred_action in {A,B,C} and pred_action!=gold_action | "
        "gold_action in {A,B,C}); denominator ToolNeededN; INVALID is audited separately"
    ),
    "UnderCall_c": "P(pred_action=NONE | gold_action=c); denominator Gold_c_N",
    "WrongCat_c": (
        "P(pred_action in {A,B,C} and pred_action!=c | gold_action=c); "
        "denominator Gold_c_N"
    ),
    "MixedCategory": (
        "P(two or more distinct routed categories in one task); denominator N"
    ),
    "FirstEnvCorrect": (
        "P(first routed environment equals gold environment | at least one routed call); "
        "denominator FirstCallN"
    ),
    "ExactToolAllowed": (
        "P(first routed tool belongs to gold tool set | at least one routed call); "
        "denominator FirstCallN"
    ),
    "FirstArgumentsValid": (
        "P(first routed call arguments pass schema/coercion validation | at least one "
        "routed call); denominator FirstCallN"
    ),
    "ToolParseFailureRate": (
        "P(task has at least one attempted tool parse failure); denominator N"
    ),
}


@dataclass(frozen=True)
class NeuronMask:
    """Validated class/layer neuron indices from the probing stage."""

    classes: dict[str, dict[int, tuple[int, ...]]]
    source_sha256: str

    def indices(self, action: str) -> dict[int, tuple[int, ...]]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action class {action!r}")
        return self.classes[action]


@dataclass(frozen=True)
class AblationCondition:
    condition_id: str
    kind: str
    masked_class: str | None
    mask_seed: int | None
    layers: dict[int, tuple[int, ...]]
    indices_sha256: str
    random_sampling: dict[str, Any] | None = None

    @property
    def selected_count(self) -> int:
        return sum(len(indices) for indices in self.layers.values())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "masked_class": self.masked_class,
            "mask_seed": self.mask_seed,
            "selected_count": self.selected_count,
            "indices_sha256": self.indices_sha256,
            "layers": {
                str(layer): [{"neuron_idx": index} for index in indices]
                for layer, indices in sorted(self.layers.items())
            },
        }
        if self.random_sampling is not None:
            payload["random_sampling"] = self.random_sampling
        return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_neuron_index(value: Any, intermediate_size: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.neuron_idx must be an integer")
    if not 0 <= value < intermediate_size:
        raise ValueError(
            f"{context}.neuron_idx={value} is outside [0, {intermediate_size})"
        )
    return value


def validate_primary_mask_contract(
    payload: Any,
    *,
    num_hidden_layers: int,
    intermediate_size: int,
    expected_rho: float = 0.003,
    expected_variant: str = "signed",
    require_train_selection: bool = True,
) -> None:
    """Audit the frozen Stage-5 primary mask beyond its structural schema."""

    if not isinstance(payload, dict):
        raise TypeError("Primary neuron mask must be an object")
    if payload.get("rho") != expected_rho:
        raise ValueError(
            f"Primary mask rho={payload.get('rho')!r}; expected {expected_rho}"
        )
    if payload.get("activation_variant") != expected_variant:
        raise ValueError(
            "Primary mask activation_variant="
            f"{payload.get('activation_variant')!r}; expected {expected_variant!r}"
        )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("Primary mask config must be an object")
    if require_train_selection:
        required_selection = {
            "selection_split": "train",
            "test_used_for_selection": False,
            "control_seed": 42,
            "topk_rule": "floor(rho * intermediate_size) independently per layer",
        }
        for key, expected in required_selection.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"Primary mask config.{key}={config.get(key)!r}; expected {expected!r}"
                )
    expected_topk = math.floor(expected_rho * intermediate_size)
    if expected_topk < 1 or config.get("topk_per_layer") != expected_topk:
        raise ValueError(
            f"Primary mask topk_per_layer must be floor(rho*I)={expected_topk}"
        )
    if payload.get("class_order") != list(ACTIONS):
        raise ValueError(f"Primary mask class_order must be {list(ACTIONS)}")
    root_layer_counts = payload.get("layer_counts")
    classes = payload.get("classes")
    if not isinstance(root_layer_counts, dict) or not isinstance(classes, dict):
        raise TypeError("Primary mask classes/layer_counts must be objects")
    if set(classes) != set(ACTIONS) or set(root_layer_counts) != set(ACTIONS):
        raise ValueError("Primary mask classes/layer_counts must cover NONE/A/B/C")
    expected_layer_keys = {str(layer) for layer in range(num_hidden_layers)}
    class_features: dict[str, set[tuple[int, int]]] = {}
    total_assignments = 0
    for action in ACTIONS:
        class_payload = classes[action]
        if not isinstance(class_payload, dict):
            raise TypeError(f"Primary mask classes.{action} must be an object")
        layers = class_payload.get("layers")
        counts = class_payload.get("layer_counts")
        if not isinstance(layers, dict) or set(layers) != expected_layer_keys:
            raise ValueError(f"Primary mask {action} must contain every model layer")
        if not isinstance(counts, dict) or set(counts) != expected_layer_keys:
            raise ValueError(f"Primary mask {action} layer_counts are incomplete")
        if root_layer_counts[action] != counts:
            raise ValueError(f"Root/class layer_counts differ for {action}")
        features: set[tuple[int, int]] = set()
        for layer_text in sorted(expected_layer_keys, key=int):
            entries = layers[layer_text]
            if not isinstance(entries, list):
                raise TypeError(f"Primary mask {action} layer {layer_text} is not a list")
            if counts[layer_text] != len(entries):
                raise ValueError(f"Primary mask {action} layer count mismatch")
            if len(entries) > expected_topk:
                raise ValueError(f"Primary mask {action} exceeds pre-difference top-k")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise TypeError("Primary mask neuron entry must be an object")
                index = _validate_neuron_index(
                    entry.get("neuron_idx"),
                    intermediate_size,
                    f"classes.{action}.layers.{layer_text}",
                )
                feature = (int(layer_text), index)
                if feature in features:
                    raise ValueError(f"Duplicate primary-mask feature for {action}")
                features.add(feature)
        if class_payload.get("total_neurons") != len(features):
            raise ValueError(f"Primary mask classes.{action}.total_neurons mismatch")
        if class_payload.get("topk_per_layer_before_set_difference") != expected_topk:
            raise ValueError(f"Primary mask {action} top-k metadata mismatch")
        class_features[action] = features
        total_assignments += len(features)
    union = set().union(*class_features.values())
    if payload.get("total_neurons") != len(union):
        raise ValueError("Primary mask root total_neurons mismatch")
    if payload.get("total_class_assignments") != total_assignments:
        raise ValueError("Primary mask total_class_assignments mismatch")
    expected_union_hash = canonical_json_sha256(
        [[layer, neuron] for layer, neuron in sorted(union)]
    )
    if payload.get("union_features_sha256") != expected_union_hash:
        raise ValueError("Primary mask union_features_sha256 mismatch")

    hashes = payload.get("hashes")
    if not isinstance(hashes, dict):
        raise TypeError("Primary mask hashes must be an object")
    required_hashes = {
        "config_sha256",
        "train_activation_sha256",
        "train_activation_manifest_sha256",
        "train_labels_sha256",
        "down_proj_column_norms_sha256",
        "runtime_provenance_sha256",
    }
    if not required_hashes <= set(hashes):
        raise ValueError(f"Primary mask hashes missing {sorted(required_hashes-set(hashes))}")
    for name in required_hashes:
        value = hashes[name]
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"Primary mask hashes.{name} is not a lowercase SHA256")
    if payload.get("runtime_provenance_sha256") != hashes["runtime_provenance_sha256"]:
        raise ValueError("Primary mask runtime provenance hashes disagree")
    project_commit = payload.get("project_git_commit")
    if not isinstance(project_commit, str) or not project_commit:
        raise ValueError("Primary mask project_git_commit must be non-empty")
    if config.get("project_git_commit") != project_commit:
        raise ValueError("Primary mask config/root project git commits disagree")

    control = payload.get("control")
    if not isinstance(control, dict) or control.get("base_seed") != 42:
        raise ValueError("Primary mask control base_seed must be 42")
    control_classes = control.get("classes")
    if not isinstance(control_classes, dict) or set(control_classes) != set(ACTIONS):
        raise ValueError("Primary mask control classes must cover NONE/A/B/C")
    for action in ACTIONS:
        item = control_classes[action]
        if not isinstance(item, dict):
            raise TypeError(f"Primary mask control.{action} must be an object")
        target_ids = item.get("target_ids")
        control_ids = item.get("control_ids")
        for side, ids in (("target", target_ids), ("control", control_ids)):
            if (
                not isinstance(ids, list)
                or not ids
                or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError(
                    f"Primary mask control {action} {side}_ids violate no-replacement"
                )
            if item.get(f"{side}_ids_sha256") != canonical_json_sha256(ids):
                raise ValueError(f"Primary mask control {action} {side} ID hash mismatch")
        if set(target_ids) & set(control_ids):
            raise ValueError(f"Primary mask control {action} target/control IDs overlap")
        if len(target_ids) != len(control_ids):
            raise ValueError(f"Primary mask control {action} sides are not equal size")
        if item.get("actual_n_each_side") != len(target_ids):
            raise ValueError(f"Primary mask control {action} actual_n_each_side mismatch")
        if classes[action].get("target_examples") != len(target_ids) or classes[
            action
        ].get("control_examples") != len(control_ids):
            raise ValueError(f"Primary mask {action} class/control example counts disagree")
        for side in ("target", "control"):
            counts = item.get(f"{side}_difficulty_counts")
            if not isinstance(counts, dict) or sum(counts.values()) != len(target_ids):
                raise ValueError(f"Primary mask control {action} difficulty counts mismatch")


def parse_neuron_mask(
    payload: Any,
    *,
    num_hidden_layers: int,
    intermediate_size: int,
    source_sha256: str = "in-memory",
    expected_model: dict[str, Any] | None = None,
) -> NeuronMask:
    """Strictly parse the cross-stage neuron-mask contract."""

    if not isinstance(payload, dict):
        raise TypeError("Neuron mask root must be an object")
    if payload.get("schema_version") != NEURON_MASK_SCHEMA_VERSION:
        raise ValueError(
            "Neuron mask schema_version must be "
            f"{NEURON_MASK_SCHEMA_VERSION!r}"
        )
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, dict) or set(raw_classes) != set(ACTIONS):
        raise ValueError(f"Neuron mask classes must be exactly {list(ACTIONS)}")
    if num_hidden_layers <= 0 or intermediate_size <= 0:
        raise ValueError("Model dimensions must be positive")
    if expected_model is not None:
        if not isinstance(expected_model, dict) or not expected_model:
            raise TypeError("expected_model must be a non-empty object")
        mask_model = payload.get("model")
        if not isinstance(mask_model, dict):
            raise TypeError("Neuron mask model metadata must be an object")
        for key, expected in expected_model.items():
            if mask_model.get(key) != expected:
                raise ValueError(
                    f"Neuron mask model.{key}={mask_model.get(key)!r}; "
                    f"expected {expected!r}"
                )

    parsed: dict[str, dict[int, tuple[int, ...]]] = {}
    for action in ACTIONS:
        class_payload = raw_classes[action]
        if not isinstance(class_payload, dict):
            raise TypeError(f"classes.{action} must be an object")
        raw_layers = class_payload.get("layers")
        if not isinstance(raw_layers, dict):
            raise TypeError(f"classes.{action}.layers must be an object")
        class_layers: dict[int, tuple[int, ...]] = {}
        for layer_text, raw_entries in raw_layers.items():
            if not isinstance(layer_text, str) or not layer_text.isdigit():
                raise TypeError(f"classes.{action}.layers keys must be decimal strings")
            layer = int(layer_text)
            if str(layer) != layer_text:
                raise ValueError(f"Non-canonical layer key {layer_text!r}")
            if not 0 <= layer < num_hidden_layers:
                raise ValueError(
                    f"classes.{action}.layers.{layer} outside "
                    f"[0, {num_hidden_layers})"
                )
            if not isinstance(raw_entries, list):
                raise TypeError(f"classes.{action}.layers.{layer} must be a list")
            indices: list[int] = []
            for entry_index, entry in enumerate(raw_entries):
                context = f"classes.{action}.layers.{layer}[{entry_index}]"
                if not isinstance(entry, dict) or "neuron_idx" not in entry:
                    raise TypeError(f"{context} must be an object containing neuron_idx")
                indices.append(
                    _validate_neuron_index(
                        entry["neuron_idx"], intermediate_size, context
                    )
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate neurons in classes.{action}.layers.{layer}")
            class_layers[layer] = tuple(sorted(indices))
        if not class_layers or not any(class_layers.values()):
            raise ValueError(f"classes.{action} selects no neurons")
        parsed[action] = dict(sorted(class_layers.items()))
    return NeuronMask(classes=parsed, source_sha256=source_sha256)


def load_neuron_mask(
    path: Path | str,
    *,
    num_hidden_layers: int,
    intermediate_size: int,
    expected_model: dict[str, Any] | None = None,
    expected_rho: float | None = None,
    expected_variant: str | None = None,
    require_train_selection: bool = False,
) -> NeuronMask:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid neuron-mask JSON: {source}") from error
    if (
        expected_rho is not None
        or expected_variant is not None
        or require_train_selection
    ):
        if expected_rho is None or expected_variant is None:
            raise ValueError(
                "Strict primary-mask loading requires expected_rho and expected_variant"
            )
        validate_primary_mask_contract(
            payload,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            expected_rho=expected_rho,
            expected_variant=expected_variant,
            require_train_selection=require_train_selection,
        )
    return parse_neuron_mask(
        payload,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        source_sha256=_sha256_path(source),
        expected_model=expected_model,
    )


def _indices_digest(layers: dict[int, tuple[int, ...]]) -> str:
    return canonical_json_sha256(
        {str(layer): list(indices) for layer, indices in sorted(layers.items())}
    )


def _random_control(
    target: dict[int, tuple[int, ...]],
    target_union: dict[int, tuple[int, ...]],
    *,
    action: str,
    seed: int,
    intermediate_size: int,
) -> tuple[dict[int, tuple[int, ...]], dict[str, Any]]:
    sampled: dict[int, tuple[int, ...]] = {}
    layer_manifests: dict[str, dict[str, Any]] = {}
    universe = range(intermediate_size)
    all_layers = sorted(set(target) | set(target_union))
    for layer in all_layers:
        target_indices = target.get(layer, ())
        excluded_indices = target_union.get(layer, ())
        target_set = set(target_indices)
        excluded_set = set(excluded_indices)
        if not target_set <= excluded_set:
            raise AssertionError("Current-class targets are absent from target union")
        complement = [index for index in universe if index not in excluded_set]
        count = len(target_indices)
        if count > len(complement):
            raise ValueError(
                f"Layer {layer} target count {count} exceeds complement "
                f"size {len(complement)}"
            )
        layer_seed_payload = f"{action}:{seed}:{layer}".encode("utf-8")
        layer_seed = int.from_bytes(hashlib.sha256(layer_seed_payload).digest()[:8], "big")
        indices = tuple(sorted(random.Random(layer_seed).sample(complement, count)))
        if set(indices) & excluded_set:
            raise AssertionError("Random control overlaps four-class target union")
        if len(indices) != count:
            raise AssertionError("Random control count differs from target")
        sampled[layer] = indices
        layer_manifests[str(layer)] = {
            "target_count": count,
            "excluded_target_union_size": len(excluded_indices),
            "excluded_target_union_indices": list(excluded_indices),
            "excluded_target_union_sha256": canonical_json_sha256(
                list(excluded_indices)
            ),
            "eligible_size": len(complement),
            "eligible_indices_sha256": canonical_json_sha256(complement),
            "sampled_indices": list(indices),
            "sampled_indices_sha256": canonical_json_sha256(list(indices)),
        }
    return sampled, {
        "selection_rule": (
            "per layer sample without replacement from "
            "range(intermediate_size) minus union(N_NONE,N_A,N_B,N_C)"
        ),
        "intermediate_size": intermediate_size,
        "action": action,
        "mask_seed": seed,
        "layers": layer_manifests,
    }


def build_ablation_conditions(
    mask: NeuronMask,
    *,
    intermediate_size: int,
    random_seeds: Sequence[int] = RANDOM_MASK_SEEDS,
) -> list[AblationCondition]:
    """Build one shared baseline, four targets, and 4x5 random controls."""

    normalized_seeds = tuple(random_seeds)
    if normalized_seeds != RANDOM_MASK_SEEDS:
        raise ValueError(f"Random mask seeds must be exactly {RANDOM_MASK_SEEDS}")
    conditions = [
        AblationCondition(
            condition_id="no_mask",
            kind="no_mask",
            masked_class=None,
            mask_seed=None,
            layers={},
            indices_sha256=_indices_digest({}),
        )
    ]
    union_layers: dict[int, tuple[int, ...]] = {}
    all_layer_ids = sorted(
        {
            layer
            for action in ACTIONS
            for layer in mask.indices(action)
        }
    )
    for layer in all_layer_ids:
        union_layers[layer] = tuple(
            sorted(
                {
                    index
                    for action in ACTIONS
                    for index in mask.indices(action).get(layer, ())
                }
            )
        )
    for action in ACTIONS:
        target = mask.indices(action)
        conditions.append(
            AblationCondition(
                condition_id=f"target_{action}",
                kind="target_mask",
                masked_class=action,
                mask_seed=None,
                layers=target,
                indices_sha256=_indices_digest(target),
            )
        )
        for seed in normalized_seeds:
            random_layers, random_sampling = _random_control(
                target,
                union_layers,
                action=action,
                seed=seed,
                intermediate_size=intermediate_size,
            )
            conditions.append(
                AblationCondition(
                    condition_id=f"random_{action}_seed{seed}",
                    kind="random_mask",
                    masked_class=action,
                    mask_seed=seed,
                    layers=random_layers,
                    indices_sha256=_indices_digest(random_layers),
                    random_sampling=random_sampling,
                )
            )
    if len(conditions) != 25 or len({item.condition_id for item in conditions}) != 25:
        raise AssertionError("Causal matrix must contain 25 unique conditions")
    return conditions


class NeuronAblationHooks(AbstractContextManager["NeuronAblationHooks"]):
    """Zero selected inputs to every Qwen3 ``mlp.down_proj`` during generation."""

    def __init__(
        self,
        layers: Sequence[Any],
        selected: dict[int, tuple[int, ...]],
        *,
        intermediate_size: int,
        require_exercised: bool = True,
    ) -> None:
        self.layers = list(layers)
        self.selected = dict(selected)
        self.intermediate_size = intermediate_size
        self.require_exercised = require_exercised
        self._handles: list[Any] = []
        self.call_counts = {layer: 0 for layer, indices in selected.items() if indices}
        self._index_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}
        for layer, indices in self.selected.items():
            if not 0 <= layer < len(self.layers):
                raise ValueError(f"Selected layer {layer} is outside model")
            if len(indices) != len(set(indices)):
                raise ValueError(f"Layer {layer} contains duplicate indices")
            for index in indices:
                _validate_neuron_index(index, intermediate_size, f"layer {layer}")

    def _hook(self, layer: int, indices: tuple[int, ...]):
        def apply_mask(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not isinstance(inputs, tuple) or not inputs:
                raise TypeError(f"Layer {layer} down_proj hook received no positional input")
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"Layer {layer} down_proj input is not a tensor")
            if hidden.ndim < 1 or hidden.shape[-1] != self.intermediate_size:
                raise ValueError(
                    f"Layer {layer} down_proj input shape {tuple(hidden.shape)} does not "
                    f"end in {self.intermediate_size}"
                )
            device_key = (layer, hidden.device.type, hidden.device.index)
            index_tensor = self._index_cache.get(device_key)
            if index_tensor is None:
                index_tensor = torch.tensor(indices, dtype=torch.long, device=hidden.device)
                self._index_cache[device_key] = index_tensor
            masked = hidden.clone()
            masked.index_fill_(-1, index_tensor, 0)
            self.call_counts[layer] += 1
            return (masked, *inputs[1:])

        return apply_mask

    def __enter__(self) -> "NeuronAblationHooks":
        if self._handles:
            raise RuntimeError("Neuron ablation hooks are already installed")
        try:
            for layer, indices in sorted(self.selected.items()):
                if not indices:
                    continue
                mlp = getattr(self.layers[layer], "mlp", None)
                down_proj = getattr(mlp, "down_proj", None)
                if down_proj is None or not hasattr(down_proj, "register_forward_pre_hook"):
                    raise TypeError(f"Layer {layer} lacks hookable mlp.down_proj")
                self._handles.append(
                    down_proj.register_forward_pre_hook(self._hook(layer, indices))
                )
        except BaseException:
            self._remove()
            raise
        return self

    def assert_exercised(self) -> None:
        missing = sorted(layer for layer, count in self.call_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"Ablation hooks were never exercised in layers {missing}")

    def _remove(self) -> None:
        errors: list[BaseException] = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as error:  # removal failure must remain visible
                errors.append(error)
        self._handles.clear()
        self._index_cache.clear()
        if errors:
            raise RuntimeError(f"Failed to remove {len(errors)} ablation hook(s)") from errors[0]

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        exercise_error: BaseException | None = None
        if exc_type is None and self.require_exercised:
            try:
                self.assert_exercised()
            except BaseException as error:
                exercise_error = error
        # Removal failure is never suppressed, including when model generation
        # also failed.  Python's chained exception context retains both errors.
        self._remove()
        if exercise_error is not None:
            raise exercise_error
        return False


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a JSON-safe rate; an undefined denominator is represented by null."""

    return float(numerator / denominator) if denominator else None


def compute_ablation_metrics(
    rows: Sequence[dict[str, Any]], *, masked_class: str | None
) -> dict[str, Any]:
    """Compute the registered action and behavioral metrics without I/O."""

    if not rows:
        raise ValueError("Cannot compute metrics for empty rows")
    if masked_class is not None and masked_class not in ACTIONS:
        raise ValueError(f"Unknown masked_class {masked_class!r}")
    gold: list[str] = []
    pred: list[str] = []
    final: list[bool] = []
    total_calls = 0
    invalid_rows = 0
    mixed_rows = 0
    parse_failure_rows = 0
    parse_failures = 0
    first_call_n = 0
    first_env_correct = 0
    exact_tool_allowed = 0
    first_arguments_valid = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Row {row_index} is not an object")
        gold_action = row.get("gold_action")
        pred_action = row.get("pred_action")
        if gold_action not in ACTIONS:
            raise ValueError(f"Row {row_index} invalid gold_action {gold_action!r}")
        if pred_action not in PREDICTIONS:
            raise ValueError(f"Row {row_index} invalid pred_action {pred_action!r}")
        if type(row.get("final_correct")) is not bool:
            raise TypeError(f"Row {row_index} final_correct must be bool")
        calls = row.get("total_tool_calls", row.get("tool_calls"))
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise TypeError(f"Row {row_index} tool-call count must be non-negative int")
        categories = row.get("tool_call_categories", [])
        if not isinstance(categories, list):
            raise TypeError(f"Row {row_index} tool_call_categories must be a list")
        invalid_count = row.get("invalid_tool_calls", 0)
        if isinstance(invalid_count, bool) or not isinstance(invalid_count, int):
            raise TypeError(f"Row {row_index} invalid_tool_calls must be int")
        mixed_value = row.get("mixed_category_calls")
        if type(mixed_value) is not bool:
            raise TypeError(f"Row {row_index} mixed_category_calls must be bool")
        derived_mixed = len(set(categories)) >= 2
        if mixed_value != derived_mixed:
            raise ValueError(
                f"Row {row_index} mixed_category_calls disagrees with routed categories"
            )
        tool_parse_failures = row.get("tool_parse_failures")
        if (
            isinstance(tool_parse_failures, bool)
            or not isinstance(tool_parse_failures, int)
            or tool_parse_failures < 0
        ):
            raise TypeError(
                f"Row {row_index} tool_parse_failures must be a non-negative int"
            )
        if calls > 0:
            observed_fields = (
                "first_env_correct",
                "exact_tool_allowed",
                "first_arguments_valid",
            )
            for field in observed_fields:
                if type(row.get(field)) is not bool:
                    raise TypeError(
                        f"Row {row_index} {field} must be bool when a call was routed"
                    )
            first_call_n += 1
            first_env_correct += int(row["first_env_correct"])
            exact_tool_allowed += int(row["exact_tool_allowed"])
            first_arguments_valid += int(row["first_arguments_valid"])
        gold.append(gold_action)
        pred.append(pred_action)
        final.append(row["final_correct"])
        total_calls += calls
        invalid_rows += int(
            pred_action == "INVALID" or invalid_count > 0 or "INVALID" in categories
        )
        mixed_rows += int(mixed_value)
        parse_failures += tool_parse_failures
        parse_failure_rows += int(tool_parse_failures > 0)

    n = len(rows)
    correct = sum(g == p for g, p in zip(gold, pred))
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for action in ACTIONS:
        tp = sum(g == action and p == action for g, p in zip(gold, pred))
        fn = sum(g == action and p != action for g, p in zip(gold, pred))
        fp = sum(g != action and p == action for g, p in zip(gold, pred))
        recall = _safe_rate(tp, tp + fn)
        if recall is None:
            raise ValueError(f"Ablation metric panel has zero gold {action} examples")
        recalls[action] = recall
        denominator = 2 * tp + fp + fn
        f1_values.append(_safe_rate(2 * tp, denominator) if denominator else 0.0)
    none_count = sum(g == "NONE" for g in gold)
    if none_count == 0:
        raise ValueError("Ablation metric panel has zero gold NONE examples")
    class_counts = {action: sum(g == action for g in gold) for action in ACTIONS}
    missing_classes = [action for action, count in class_counts.items() if count == 0]
    if missing_classes:
        raise ValueError(
            f"Ablation metric panel has zero denominator for classes {missing_classes}"
        )
    tool_needed_n = n - none_count
    if tool_needed_n == 0:
        raise ValueError("Ablation metric panel has zero tool-needed examples")
    overcalls = sum(g == "NONE" and p != "NONE" for g, p in zip(gold, pred))
    under_calls = sum(
        g != "NONE" and p == "NONE" for g, p in zip(gold, pred)
    )
    wrong_categories = sum(
        g != "NONE" and p in {"A", "B", "C"} and p != g
        for g, p in zip(gold, pred)
    )
    off_target_indices = (
        [index for index, value in enumerate(gold) if value != masked_class]
        if masked_class is not None
        else []
    )
    off_target_errors = sum(gold[index] != pred[index] for index in off_target_indices)
    metrics = {
        "N": n,
        "ActionAcc": float(correct / n),
        "MacroF1_action": float(np.mean(f1_values)),
        **{f"Recall_{action}": recalls[action] for action in ACTIONS},
        "FinalAcc": float(sum(final) / n),
        "TotalTC": total_calls,
        "AvgTC": float(total_calls / n),
        "OverCall": _safe_rate(overcalls, none_count),
        "Gold_NONE_N": none_count,
        "ToolNeededN": tool_needed_n,
        "UnderCall": _safe_rate(under_calls, tool_needed_n),
        "WrongCat": _safe_rate(wrong_categories, tool_needed_n),
        "InvalidRate": float(invalid_rows / n),
        "Mixed": float(mixed_rows / n),
        "MixedCategory": float(mixed_rows / n),
        "ToolParseFailures": parse_failures,
        "ToolParseFailureRows": parse_failure_rows,
        "ToolParseFailureRate": float(parse_failure_rows / n),
        "FirstCallN": first_call_n,
        "FirstEnvCorrect": (
            _safe_rate(first_env_correct, first_call_n) if first_call_n else None
        ),
        "ExactEnv": (
            _safe_rate(first_env_correct, first_call_n) if first_call_n else None
        ),
        "ExactToolAllowed": (
            _safe_rate(exact_tool_allowed, first_call_n) if first_call_n else None
        ),
        "ExactTool": (
            _safe_rate(exact_tool_allowed, first_call_n) if first_call_n else None
        ),
        "FirstArgumentsValid": (
            _safe_rate(first_arguments_valid, first_call_n) if first_call_n else None
        ),
        "InvalidArgumentRate": (
            _safe_rate(first_call_n - first_arguments_valid, first_call_n)
            if first_call_n
            else None
        ),
        "OffTargetN": len(off_target_indices),
        "OffTargetError": (
            _safe_rate(off_target_errors, len(off_target_indices))
            if masked_class is not None
            else None
        ),
    }
    for action in ("A", "B", "C"):
        action_n = class_counts[action]
        action_under = sum(
            gold_value == action and pred_value == "NONE"
            for gold_value, pred_value in zip(gold, pred)
        )
        action_wrong = sum(
            gold_value == action
            and pred_value in {"A", "B", "C"}
            and pred_value != action
            for gold_value, pred_value in zip(gold, pred)
        )
        metrics[f"Gold_{action}_N"] = action_n
        metrics[f"UnderCall_{action}"] = _safe_rate(action_under, action_n)
        metrics[f"WrongCat_{action}"] = _safe_rate(action_wrong, action_n)
    return metrics


def _numeric_metric_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    excluded = {
        "condition_id",
        "kind",
        "masked_class",
        "mask_seed",
        "generation_seed",
        "indices_sha256",
    }
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded
            and (
                value is None
                or (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
            )
        }
    )


def summarize_condition_metrics(
    metric_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not metric_rows:
        raise ValueError("No per-condition metric rows")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[str(row["condition_id"])].append(row)
    metric_names = _numeric_metric_names(metric_rows)
    summaries: list[dict[str, Any]] = []
    for condition_id, rows in sorted(groups.items()):
        first = rows[0]
        summary: dict[str, Any] = {
            "condition_id": condition_id,
            "kind": first["kind"],
            "masked_class": first.get("masked_class"),
            "mask_seed": first.get("mask_seed"),
            "n_generation_seeds": len(rows),
            "indices_sha256": first["indices_sha256"],
        }
        if any(
            row.get("indices_sha256") != first["indices_sha256"]
            or row.get("kind") != first["kind"]
            or row.get("masked_class") != first.get("masked_class")
            or row.get("mask_seed") != first.get("mask_seed")
            for row in rows
        ):
            raise ValueError(f"Condition metadata varies for {condition_id}")
        for metric in metric_names:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in rows
                    if isinstance(row.get(metric), (int, float))
                    and not isinstance(row.get(metric), bool)
                ],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(finite.mean()) if finite.size else None
            summary[f"{metric}_population_sd"] = (
                float(finite.std(ddof=0)) if finite.size else None
            )
        summaries.append(summary)
    return summaries


def build_causal_criteria(
    metric_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the preregistered effect-size screen; this is not significance."""

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_id[str(row["condition_id"])].append(row)
    baseline_by_seed = {
        int(row["generation_seed"]): row for row in by_id.get("no_mask", [])
    }
    criteria: list[dict[str, Any]] = []
    for action in ACTIONS:
        target_rows = by_id.get(f"target_{action}", [])
        random_groups = [by_id.get(f"random_{action}_seed{seed}", []) for seed in RANDOM_MASK_SEEDS]
        expected_generation_seeds = set(baseline_by_seed)
        complete = bool(expected_generation_seeds) and all(
            {int(row["generation_seed"]) for row in group} == expected_generation_seeds
            for group in [target_rows, *random_groups]
        )
        if not complete:
            criteria.append(
                {
                    "masked_class": action,
                    "metric": f"Recall_{action}",
                    "status": "insufficient_conditions",
                    "criterion_met": None,
                    "interpretation": "effect-size screening criterion; not statistical significance",
                }
            )
            continue

        metric = f"Recall_{action}"

        def mean_paired_drop(rows: Sequence[dict[str, Any]]) -> float:
            by_seed = {int(row["generation_seed"]): row for row in rows}
            drops = [
                float(baseline_by_seed[seed][metric]) - float(by_seed[seed][metric])
                for seed in sorted(expected_generation_seeds)
            ]
            return float(np.mean(drops))

        target_drop = mean_paired_drop(target_rows)
        random_drops = np.asarray(
            [mean_paired_drop(group) for group in random_groups], dtype=float
        )
        random_mean = float(random_drops.mean())
        random_sd = float(random_drops.std(ddof=0))
        threshold = random_mean + random_sd
        criteria.append(
            {
                "masked_class": action,
                "metric": metric,
                "status": "complete",
                "target_recall_degradation": target_drop,
                "random_recall_degradations": random_drops.tolist(),
                "random_mean_degradation": random_mean,
                "random_population_sd": random_sd,
                "criterion_threshold": threshold,
                "criterion_met": bool(target_drop > threshold),
                "interpretation": "effect-size screening criterion; not statistical significance",
            }
        )
    return criteria


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
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


def _confusion(rows: Iterable[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(ACTIONS), len(PREDICTIONS)), dtype=int)
    gold_index = {action: index for index, action in enumerate(ACTIONS)}
    pred_index = {action: index for index, action in enumerate(PREDICTIONS)}
    for row in rows:
        matrix[gold_index[row["gold_action"]], pred_index[row["pred_action"]]] += 1
    return matrix


def _plot_reports(
    checkpoints: Sequence[dict[str, Any]],
    criteria: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [row for row in criteria if row.get("status") == "complete"]
    if len(complete) != len(ACTIONS):
        return

    labels = [row["masked_class"] for row in complete]
    target = np.asarray([row["target_recall_degradation"] for row in complete])
    random_mean = np.asarray([row["random_mean_degradation"] for row in complete])
    random_sd = np.asarray([row["random_population_sd"] for row in complete])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, target, width, label="target")
    ax.bar(
        x + width / 2,
        random_mean,
        width,
        yerr=random_sd,
        capsize=4,
        label="random mean ± population SD",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Paired target-class recall degradation")
    ax.set_xlabel("Masked action class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recall_drop.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(random_mean, target, xerr=random_sd, fmt="o", capsize=4)
    low = float(min(np.min(random_mean - random_sd), np.min(target), 0.0))
    high = float(max(np.max(random_mean + random_sd), np.max(target), 0.0))
    padding = max((high - low) * 0.08, 0.01)
    ax.plot([low - padding, high + padding], [low - padding, high + padding], "--")
    for label, x_value, y_value in zip(labels, random_mean, target):
        ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Random mean recall degradation")
    ax.set_ylabel("Target recall degradation")
    fig.tight_layout()
    fig.savefig(output_dir / "target_vs_random.png", dpi=180)
    plt.close(fig)

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints:
        by_condition[checkpoint["condition"]["condition_id"]].extend(checkpoint["rows"])
    figure, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)
    condition_ids = ["no_mask", *(f"target_{action}" for action in ACTIONS)]
    image = None
    for axis, condition_id in zip(axes, condition_ids):
        counts = _confusion(by_condition[condition_id]).astype(float)
        denominators = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts),
            where=denominators != 0,
        )
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(condition_id)
        axis.set_xticks(range(len(PREDICTIONS)), PREDICTIONS, rotation=45)
        axis.set_yticks(range(len(ACTIONS)), ACTIONS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Gold")
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.8)
    figure.savefig(output_dir / "confusion_before_after.png", dpi=180)
    plt.close(figure)


def write_ablation_reports(
    checkpoints: Sequence[dict[str, Any]],
    output_dir: Path | str,
    *,
    expected_condition_ids: Sequence[str],
    expected_generation_seeds: Sequence[int],
) -> dict[str, Any]:
    """Atomically refresh metrics and create figures once the matrix is complete."""

    if not checkpoints:
        raise ValueError("No ablation checkpoints to summarize")
    destination = Path(output_dir).resolve()
    metric_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for checkpoint in checkpoints:
        if checkpoint.get("schema_version") != ABLATION_SCHEMA_VERSION:
            raise ValueError("Unexpected ablation checkpoint schema")
        condition = checkpoint.get("condition")
        metrics = checkpoint.get("metrics")
        if not isinstance(condition, dict) or not isinstance(metrics, dict):
            raise TypeError("Malformed ablation checkpoint")
        pair = (condition["condition_id"], int(checkpoint["generation_seed"]))
        if pair in seen_pairs:
            raise ValueError(f"Duplicate ablation checkpoint {pair}")
        seen_pairs.add(pair)
        metric_rows.append(
            {
                "condition_id": condition["condition_id"],
                "kind": condition["kind"],
                "masked_class": condition.get("masked_class"),
                "mask_seed": condition.get("mask_seed"),
                "generation_seed": int(checkpoint["generation_seed"]),
                "indices_sha256": condition["indices_sha256"],
                **metrics,
            }
        )
    expected_pairs = {
        (condition_id, int(seed))
        for condition_id in expected_condition_ids
        for seed in expected_generation_seeds
    }
    extra = seen_pairs - expected_pairs
    if extra:
        raise ValueError(f"Unexpected seed/condition checkpoints: {sorted(extra)}")
    missing = expected_pairs - seen_pairs
    summaries = summarize_condition_metrics(metric_rows)
    criteria = build_causal_criteria(metric_rows)
    summary = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "panel_status": "complete" if not missing else "partial",
        "expected_condition_count": len(expected_condition_ids),
        "expected_generation_seeds": list(expected_generation_seeds),
        "completed_seed_condition_count": len(seen_pairs),
        "expected_seed_condition_count": len(expected_pairs),
        "missing_seed_conditions": [
            {"condition_id": condition, "generation_seed": seed}
            for condition, seed in sorted(missing)
        ],
        "criteria_are_statistical_significance_tests": False,
        "metric_contract": ABLATION_METRIC_CONTRACT,
        "causal_screening_criteria": criteria,
        "condition_summaries": summaries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(destination / "per_condition_metrics.csv", metric_rows)
    _atomic_write_csv(destination / "summary.csv", summaries)
    atomic_write_json(destination / "summary.json", summary, overwrite=True)
    if not missing:
        _plot_reports(checkpoints, criteria, destination)
    return summary
