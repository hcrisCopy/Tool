"""Neuron-row-masked MLP LoRA and assistant-only SFT tokenization.

``peft`` is deliberately imported only by :func:`attach_masked_lora`; artifact
inspection, mask construction, and unit tests therefore do not require that
optional training dependency to be installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .constants import ACTIONS, EXPECTED_TOOL_COUNT
from .io_utils import canonical_json_sha256, sha256_bytes
from .neuron_ablation import NEURON_MASK_SCHEMA_VERSION, NeuronMask
from .sft import SFT_MANIFEST_SCHEMA_VERSION, SFT_SCHEMA_VERSION


MASKED_LORA_SCHEMA_VERSION = "when2tool-masked-lora-v1"
EXPECTED_PEFT_VERSION = "0.17.1"
EXPECTED_TRANSFORMERS_VERSION = "4.55.2"
LORA_TARGET_MODULES = ("gate_proj", "up_proj")
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TRAINING_SEED = 42
RANDOM_LORA_SEEDS = (0, 1, 2)
MAX_LENGTH = 8192
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)(?<![a-z0-9_])[a-z]:[\\/]")
_INTERNAL_POSIX_RE = re.compile(
    r"(?<![a-zA-Z0-9_.-])/(?:root|home|mnt|workspace)(?:/|\b)"
)
_JSON_METADATA_FILES = frozenset(
    {
        "adapter_config.json",
        "generation_config.json",
        "mask_snapshot.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "trainer_state.json",
    }
)
_TEXT_METADATA_FILES = frozenset({"README.md"})
_EXCLUDED_TOKENIZER_PAYLOAD_FILES = frozenset(
    {"tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"}
)
_TARGET_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>[0-9]+)\.mlp\."
    r"(?P<projection>gate_proj|up_proj)\Z"
)


@dataclass(frozen=True)
class RowSelection:
    mode: str
    random_seed: int | None
    layers: dict[int, tuple[int, ...]]
    indices_sha256: str
    target_union_sha256: str
    target_union_layers: dict[int, tuple[int, ...]]
    num_hidden_layers: int
    intermediate_size: int

    @property
    def selected_unique_rows(self) -> int:
        return sum(len(indices) for indices in self.layers.values())

    def snapshot(self) -> dict[str, Any]:
        if self.mode == "dense":
            layers: dict[str, Any] = {
                str(layer): {
                    "selection": "all_rows",
                    "count": len(indices),
                }
                for layer, indices in sorted(self.layers.items())
            }
        else:
            layers = {
                str(layer): {
                    "selection": "explicit_indices",
                    "count": len(indices),
                    "indices": list(indices),
                }
                for layer, indices in sorted(self.layers.items())
            }
            if self.mode == "random":
                for layer_text, item in layers.items():
                    layer = int(layer_text)
                    target = self.target_union_layers[layer]
                    target_set = set(target)
                    eligible = [
                        index
                        for index in range(self.intermediate_size)
                        if index not in target_set
                    ]
                    item.update(
                        {
                            "target_union_count": len(target),
                            "target_union_indices_sha256": canonical_json_sha256(
                                list(target)
                            ),
                            "eligible_complement_count": len(eligible),
                            "eligible_complement_indices_sha256": canonical_json_sha256(
                                eligible
                            ),
                        }
                    )
        return {
            "schema_version": MASKED_LORA_SCHEMA_VERSION,
            "mode": self.mode,
            "random_seed": self.random_seed,
            "sampling": (
                "all gate/up output rows"
                if self.mode == "dense"
                else (
                    "A/B/C/NONE target union"
                    if self.mode == "target"
                    else "per-layer same-count sample from strict complement of A/B/C/NONE union"
                )
            ),
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "selected_unique_rows": self.selected_unique_rows,
            "indices_sha256": self.indices_sha256,
            "target_union_sha256": self.target_union_sha256,
            "layers": layers,
        }


@dataclass
class LoraBMaskEntry:
    module_name: str
    layer: int
    projection: str
    weight: torch.nn.Parameter
    selected_indices: tuple[int, ...]
    row_mask: torch.Tensor | None
    hook: Any | None


@dataclass
class MaskedLoraController:
    selection: RowSelection
    adapter_name: str
    entries: list[LoraBMaskEntry]

    def remove_hooks(self) -> None:
        for entry in self.entries:
            if entry.hook is not None:
                entry.hook.remove()
                entry.hook = None

    def assert_nonselected_rows_zero(self) -> None:
        """Fail if optimizer updates escaped any selected LoRA-B row."""

        if self.selection.mode == "dense":
            return
        for entry in self.entries:
            selected = set(entry.selected_indices)
            nonselected = [
                index for index in range(entry.weight.shape[0]) if index not in selected
            ]
            if not nonselected:
                continue
            rows = entry.weight.detach().index_select(
                0, torch.tensor(nonselected, device=entry.weight.device)
            )
            if torch.count_nonzero(rows).item() != 0:
                raise AssertionError(
                    f"Non-selected LoRA-B rows changed in {entry.module_name}"
                )

    def assert_masked_gradients_zero(self) -> None:
        """Test/debug assertion to run after backward and before zero_grad."""

        if self.selection.mode == "dense":
            return
        for entry in self.entries:
            gradient = entry.weight.grad
            if gradient is None:
                raise AssertionError(f"Missing LoRA-B gradient for {entry.module_name}")
            assert entry.row_mask is not None
            forbidden = gradient.masked_select(
                ~entry.row_mask.to(device=gradient.device)
            )
            if torch.count_nonzero(forbidden).item() != 0:
                raise AssertionError(
                    f"Non-selected LoRA-B gradient is nonzero in {entry.module_name}"
                )


def _indices_digest(layers: Mapping[int, Sequence[int]]) -> str:
    return canonical_json_sha256(
        {str(layer): list(indices) for layer, indices in sorted(layers.items())}
    )


def target_union_layers(
    mask: NeuronMask,
    *,
    num_hidden_layers: int,
    intermediate_size: int,
) -> dict[int, tuple[int, ...]]:
    """Union all four classes per layer, retaining explicit empty layers."""

    if num_hidden_layers <= 0 or intermediate_size <= 0:
        raise ValueError("Model dimensions must be positive")
    if set(mask.classes) != set(ACTIONS):
        raise ValueError(
            f"Neuron mask must contain all actions; missing={sorted(set(ACTIONS) - set(mask.classes))}"
        )
    output: dict[int, tuple[int, ...]] = {}
    action_totals = {action: 0 for action in ACTIONS}
    for layer in range(num_hidden_layers):
        union: set[int] = set()
        for action in ACTIONS:
            indices = mask.indices(action).get(layer, ())
            action_totals[action] += len(indices)
            for index in indices:
                if not 0 <= index < intermediate_size:
                    raise ValueError(
                        f"Mask {action} layer {layer} index {index} is out of range"
                    )
                union.add(index)
        output[layer] = tuple(sorted(union))
    missing = [action for action, count in action_totals.items() if count == 0]
    if missing:
        raise ValueError(f"Neuron mask actions select no rows: {missing}")
    if not any(output.values()):
        raise ValueError("Four-action target union is empty")
    return output


def build_row_selection(
    mask: NeuronMask,
    *,
    mode: str,
    num_hidden_layers: int,
    intermediate_size: int,
    random_seed: int | None = None,
) -> RowSelection:
    """Build target, strict-complement random, or dense row selections."""

    if mode not in {"target", "random", "dense"}:
        raise ValueError("mode must be target, random, or dense")
    union = target_union_layers(
        mask,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
    )
    union_sha = _indices_digest(union)
    if mode == "target":
        if random_seed is not None:
            raise ValueError("target mode does not accept random_seed")
        layers = union
    elif mode == "dense":
        if random_seed is not None:
            raise ValueError("dense mode does not accept random_seed")
        all_rows = tuple(range(intermediate_size))
        layers = {layer: all_rows for layer in range(num_hidden_layers)}
    else:
        if random_seed not in RANDOM_LORA_SEEDS:
            raise ValueError(f"random mode seed must be one of {RANDOM_LORA_SEEDS}")
        layers = {}
        for layer, target_indices in union.items():
            target_set = set(target_indices)
            complement = [
                index for index in range(intermediate_size) if index not in target_set
            ]
            count = len(target_indices)
            if count > len(complement):
                raise ValueError(
                    f"Layer {layer} union count {count} exceeds complement size {len(complement)}"
                )
            seed_payload = f"when2tool-random-union-v1:{random_seed}:{layer}".encode(
                "utf-8"
            )
            layer_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:8], "big"
            )
            sampled = tuple(sorted(random.Random(layer_seed).sample(complement, count)))
            if set(sampled) & target_set:
                raise AssertionError("Random LoRA selection overlaps four-action union")
            if len(sampled) != count:
                raise AssertionError(
                    "Random LoRA layer count differs from target union"
                )
            layers[layer] = sampled
    return RowSelection(
        mode=mode,
        random_seed=random_seed,
        layers=layers,
        indices_sha256=_indices_digest(layers),
        target_union_sha256=union_sha,
        target_union_layers=union,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
    )


def validate_probe_mask_metadata(
    payload: Any, selection: RowSelection
) -> dict[str, Any]:
    """Bind the parsed rows back to the probing artifact's union receipt."""

    if not isinstance(payload, dict):
        raise TypeError("Neuron mask artifact must be an object")
    if payload.get("schema_version") != NEURON_MASK_SCHEMA_VERSION:
        raise ValueError("Neuron mask schema differs from probing/ablation v1")
    if payload.get("class_order") != list(ACTIONS):
        raise ValueError("Neuron mask class_order differs from the frozen taxonomy")
    features = [
        [layer, neuron]
        for layer, indices in sorted(selection.target_union_layers.items())
        for neuron in indices
    ]
    expected_union_sha256 = canonical_json_sha256(features)
    if payload.get("union_features_sha256") != expected_union_sha256:
        raise ValueError("Neuron mask union_features_sha256 does not match its classes")
    if payload.get("total_neurons") != len(features):
        raise ValueError("Neuron mask total_neurons does not match its class union")
    classes = payload.get("classes")
    if not isinstance(classes, dict) or set(classes) != set(ACTIONS):
        raise ValueError("Neuron mask classes must be exactly NONE/A/B/C")
    assignments = 0
    expected_layers = {str(layer) for layer in range(selection.num_hidden_layers)}
    for action in ACTIONS:
        class_payload = classes[action]
        layers = (
            class_payload.get("layers") if isinstance(class_payload, dict) else None
        )
        counts = (
            class_payload.get("layer_counts")
            if isinstance(class_payload, dict)
            else None
        )
        if not isinstance(layers, dict) or set(layers) != expected_layers:
            raise ValueError(f"Neuron mask {action} must enumerate every model layer")
        actual_counts = {
            layer: len(entries) if isinstance(entries, list) else -1
            for layer, entries in layers.items()
        }
        if counts != actual_counts:
            raise ValueError(f"Neuron mask {action} layer_counts are inconsistent")
        total = sum(actual_counts.values())
        if class_payload.get("total_neurons") != total:
            raise ValueError(f"Neuron mask {action} total_neurons is inconsistent")
        assignments += total
    if payload.get("total_class_assignments") != assignments:
        raise ValueError("Neuron mask total_class_assignments is inconsistent")
    return {
        "union_features_sha256": expected_union_sha256,
        "total_union_neurons": len(features),
        "total_class_assignments": assignments,
    }


def _mapping_keys(value: Any, context: str) -> set[str]:
    if not hasattr(value, "keys") or not hasattr(value, "__getitem__"):
        raise TypeError(f"{context} must be a ModuleDict-like mapping")
    keys = set(value.keys())
    if not keys or not all(isinstance(key, str) and key for key in keys):
        raise ValueError(f"{context} has no valid adapter keys")
    return keys


def _discover_target_modules(
    model: torch.nn.Module, num_hidden_layers: int
) -> dict[tuple[int, str], tuple[str, Any]]:
    found: dict[tuple[int, str], tuple[str, Any]] = {}
    for name, module in model.named_modules():
        match = _TARGET_RE.search(name)
        if match is None:
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            # The base Linear may match on an unwrapped/fake model.  It is not a
            # valid training target until PEFT has attached both matrices.
            continue
        layer = int(match.group("layer"))
        projection = match.group("projection")
        key = (layer, projection)
        if key in found:
            raise ValueError(f"Duplicate LoRA target for layer/projection {key}")
        found[key] = (name, module)
    expected = {
        (layer, projection)
        for layer in range(num_hidden_layers)
        for projection in LORA_TARGET_MODULES
    }
    if set(found) != expected:
        raise ValueError(
            "LoRA target topology differs from gate/up for every layer: "
            f"missing={sorted(expected - set(found))[:20]}, "
            f"extra={sorted(set(found) - expected)[:20]}"
        )
    return found


def _resolve_adapter_name(
    modules: Mapping[tuple[int, str], tuple[str, Any]], requested: str | None
) -> str:
    common: set[str] | None = None
    for _key, (name, module) in modules.items():
        a_keys = _mapping_keys(module.lora_A, f"{name}.lora_A")
        b_keys = _mapping_keys(module.lora_B, f"{name}.lora_B")
        if a_keys != b_keys:
            raise ValueError(f"{name} LoRA-A/B adapter keys differ")
        common = a_keys if common is None else common & a_keys
    assert common is not None
    if requested is not None:
        if requested not in common:
            raise ValueError(f"Requested adapter {requested!r} is absent from a target")
        return requested
    if len(common) != 1:
        raise ValueError(
            f"Expected exactly one shared LoRA adapter, got {sorted(common)}"
        )
    return next(iter(common))


def register_lora_b_gradient_masks(
    model: torch.nn.Module,
    selection: RowSelection,
    *,
    adapter_name: str | None = None,
) -> MaskedLoraController:
    """Register row hooks on LoRA-B; LoRA-A remains fully trainable."""

    modules = _discover_target_modules(model, selection.num_hidden_layers)
    adapter = _resolve_adapter_name(modules, adapter_name)
    entries: list[LoraBMaskEntry] = []
    for (layer, projection), (name, module) in sorted(modules.items()):
        a_module = module.lora_A[adapter]
        b_module = module.lora_B[adapter]
        a_weight = getattr(a_module, "weight", None)
        b_weight = getattr(b_module, "weight", None)
        if not isinstance(a_weight, torch.nn.Parameter) or not isinstance(
            b_weight, torch.nn.Parameter
        ):
            raise TypeError(f"{name} LoRA A/B must expose Parameter weights")
        if a_weight.ndim != 2 or b_weight.ndim != 2:
            raise ValueError(f"{name} LoRA A/B weights must be rank 2")
        if a_weight.shape[0] != LORA_RANK or b_weight.shape[1] != LORA_RANK:
            raise ValueError(f"{name} LoRA rank is not {LORA_RANK}")
        if b_weight.shape[0] != selection.intermediate_size:
            raise ValueError(
                f"{name} LoRA-B output rows={b_weight.shape[0]}, "
                f"expected {selection.intermediate_size}"
            )
        if not a_weight.requires_grad or not b_weight.requires_grad:
            raise ValueError(f"{name} LoRA A/B must both remain trainable")
        selected = selection.layers[layer]
        if selection.mode == "dense":
            row_mask = None
            hook = None
        else:
            row_mask = torch.zeros(
                (selection.intermediate_size, 1),
                dtype=torch.bool,
                device=b_weight.device,
            )
            if selected:
                index_tensor = torch.tensor(selected, device=b_weight.device)
                row_mask[index_tensor] = True
            nonselected = row_mask[:, 0] == 0
            if torch.count_nonzero(b_weight.detach()[nonselected]).item() != 0:
                raise ValueError(
                    f"{name} non-selected LoRA-B rows are nonzero before training"
                )

            def mask_gradient(
                gradient: torch.Tensor, *, captured_mask: torch.Tensor = row_mask
            ) -> torch.Tensor:
                return gradient.masked_fill(
                    ~captured_mask.to(device=gradient.device), 0
                )

            hook = b_weight.register_hook(mask_gradient)
        entries.append(
            LoraBMaskEntry(
                module_name=name,
                layer=layer,
                projection=projection,
                weight=b_weight,
                selected_indices=selected,
                row_mask=row_mask,
                hook=hook,
            )
        )
    controller = MaskedLoraController(selection, adapter, entries)
    controller.assert_nonselected_rows_zero()
    return controller


def validate_training_dependency_versions() -> dict[str, str]:
    """Fail before model loading unless the frozen PEFT/Transformers pair is active."""

    versions: dict[str, str] = {}
    for package, expected in (
        ("peft", EXPECTED_PEFT_VERSION),
        ("transformers", EXPECTED_TRANSFORMERS_VERSION),
    ):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Masked-LoRA training requires {package}=={expected}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"Masked-LoRA runtime requires {package}=={expected}, got {actual}"
            )
        versions[package] = actual
    return versions


def _import_peft() -> tuple[Any, Any, Any]:
    validate_training_dependency_versions()
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError(
            "Masked-LoRA training requires the optional 'peft' package in the "
            "remote training environment"
        ) from error
    return LoraConfig, TaskType, get_peft_model


def normalize_peft_base_model_identity(
    model: torch.nn.Module, base_model_slug: str
) -> tuple[str, ...]:
    """Replace local PEFT base paths with the registered portable model ID."""

    if not isinstance(base_model_slug, str) or not base_model_slug.strip():
        raise ValueError("base_model_slug must be non-empty")
    if Path(base_model_slug).is_absolute() or _WINDOWS_ABSOLUTE_RE.search(
        base_model_slug
    ):
        raise ValueError("base_model_slug must be a portable model ID, not a path")
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or not configs:
        raise TypeError("PEFT model must expose a non-empty peft_config mapping")
    normalized: list[str] = []
    for adapter_name, peft_config in configs.items():
        if not isinstance(adapter_name, str) or not adapter_name:
            raise ValueError("PEFT adapter names must be non-empty strings")
        if not hasattr(peft_config, "base_model_name_or_path"):
            raise TypeError(
                f"PEFT config {adapter_name!r} lacks base_model_name_or_path"
            )
        peft_config.base_model_name_or_path = base_model_slug
        if peft_config.base_model_name_or_path != base_model_slug:
            raise AssertionError(
                "PEFT base model identity normalization did not persist"
            )
        normalized.append(adapter_name)
    return tuple(sorted(normalized))


def normalize_tokenizer_model_identity(tokenizer: Any, base_model_slug: str) -> None:
    """Prevent tokenizer metadata from retaining its local load directory."""

    tokenizer.name_or_path = base_model_slug
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs["name_or_path"] = base_model_slug


def audit_portable_adapter_metadata(
    adapter_dir: Path | str, *, forbidden_paths: Sequence[Path | str]
) -> dict[str, Any]:
    """Fail if published text metadata discloses an internal absolute path."""

    root = Path(adapter_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    forbidden: set[str] = set()
    for value in forbidden_paths:
        resolved = Path(value).resolve()
        forbidden.update({str(resolved), resolved.as_posix()})
    forbidden.discard("")
    scanned: list[str] = []
    excluded_payloads: list[str] = []

    def reject_internal_path(value: str, context: str) -> None:
        if _WINDOWS_ABSOLUTE_RE.search(value) or _INTERNAL_POSIX_RE.search(value):
            raise ValueError(
                f"Adapter text metadata contains an internal absolute path: {context}"
            )
        if any(candidate in value for candidate in forbidden):
            raise ValueError(
                f"Adapter text metadata contains a resolved project path: {context}"
            )

    def inspect_json(value: Any, context: str, *, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                inspect_json(child, f"{context}.{child_key}", key=str(child_key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect_json(child, f"{context}[{index}]", key=key)
        elif isinstance(value, str):
            path_key = key is not None and any(
                marker in key.lower()
                for marker in ("path", "repo", "directory", "filename")
            )
            has_absolute_prefix = bool(
                _WINDOWS_ABSOLUTE_RE.search(value)
                or _INTERNAL_POSIX_RE.search(value)
                or any(candidate in value for candidate in forbidden)
            )
            if path_key or has_absolute_prefix:
                reject_internal_path(value, context)

    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name in _EXCLUDED_TOKENIZER_PAYLOAD_FILES:
            excluded_payloads.append(relative)
            continue
        if path.name not in _JSON_METADATA_FILES | _TEXT_METADATA_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Adapter metadata is not UTF-8: {relative}") from error
        if path.name in _JSON_METADATA_FILES:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid adapter metadata JSON: {relative}"
                ) from error
            inspect_json(payload, relative)
        else:
            reject_internal_path(text, relative)
        scanned.append(relative)
    if "adapter_config.json" not in scanned:
        raise FileNotFoundError(root / "adapter_config.json")
    return {
        "policy": "no-internal-absolute-paths-v1",
        "scanned_files": scanned,
        "scanned_files_sha256": canonical_json_sha256(scanned),
        "excluded_tokenizer_payload_files": excluded_payloads,
    }


def attach_masked_lora(
    model: torch.nn.Module,
    selection: RowSelection,
) -> tuple[torch.nn.Module, MaskedLoraController, dict[str, Any]]:
    """Attach the fixed gate/up rank-8 adapter and apply the selected row mask."""

    LoraConfig, TaskType, get_peft_model = _import_peft()
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, lora_config)
    controller = register_lora_b_gradient_masks(peft_model, selection)
    accounting = parameter_accounting(peft_model, controller)
    return peft_model, controller, accounting


def parameter_accounting(
    model: torch.nn.Module, controller: MaskedLoraController
) -> dict[str, Any]:
    """Report raw PEFT parameters separately from row-mask-effective ones."""

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise ValueError("Model has no trainable LoRA parameters")
    unexpected = [
        name for name in trainable if "lora_A" not in name and "lora_B" not in name
    ]
    if unexpected:
        raise ValueError(f"Non-LoRA parameters are trainable: {unexpected[:20]}")
    raw = sum(parameter.numel() for parameter in trainable.values())
    lora_a = sum(
        parameter.numel() for name, parameter in trainable.items() if "lora_A" in name
    )
    lora_b_raw = sum(
        parameter.numel() for name, parameter in trainable.items() if "lora_B" in name
    )
    controlled_b_ids = {id(entry.weight) for entry in controller.entries}
    trainable_b_ids = {
        id(parameter) for name, parameter in trainable.items() if "lora_B" in name
    }
    if controlled_b_ids != trainable_b_ids:
        raise ValueError(
            "Controlled LoRA-B matrices differ from trainable LoRA-B matrices"
        )
    effective_b = sum(
        len(entry.selected_indices) * int(entry.weight.shape[1])
        for entry in controller.entries
    )
    if controller.selection.mode == "dense" and effective_b != lora_b_raw:
        raise AssertionError("Dense LoRA effective-B count differs from raw-B count")
    effective = lora_a + effective_b
    return {
        "raw_trainable_parameters": raw,
        "effective_trainable_parameters": effective,
        "lora_a_parameters_fully_trainable": lora_a,
        "lora_b_parameters_raw": lora_b_raw,
        "lora_b_parameters_effective": effective_b,
        "raw_lora_b_output_rows": sum(
            int(entry.weight.shape[0]) for entry in controller.entries
        ),
        "effective_selected_lora_b_output_rows": sum(
            len(entry.selected_indices) for entry in controller.entries
        ),
        "selected_unique_neuron_rows": controller.selection.selected_unique_rows,
        "note": (
            "LoRA-A is fully trainable. Effective counts remove masked LoRA-B "
            "rows only and must not be reported as raw trainable parameters."
        ),
    }


def load_sft_records(
    data_path: Path | str,
    manifest_path: Path | str,
    *,
    expected_full_menu_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a JSONL only when its manifest and per-row hashes are intact."""

    data_source = Path(data_path).resolve()
    manifest_source = Path(manifest_path).resolve()
    if not data_source.is_file():
        raise FileNotFoundError(data_source)
    if not manifest_source.is_file():
        raise FileNotFoundError(manifest_source)
    payload = data_source.read_bytes()
    try:
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid SFT manifest {manifest_source}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SFT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Unexpected SFT manifest schema")
    if manifest.get("full_menu_sha256") != expected_full_menu_sha256:
        raise ValueError("SFT manifest full-menu hash differs from canonical menu")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise TypeError("SFT manifest lacks output receipt")
    if output.get("file") != data_source.name:
        raise ValueError("SFT manifest output filename differs from JSONL")
    if output.get("sha256") != sha256_bytes(payload) or output.get("bytes") != len(
        payload
    ):
        raise ValueError("SFT JSONL hash/size differs from its manifest")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"Blank JSONL line {line_number}")
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid SFT JSONL line {line_number}") from error
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != SFT_SCHEMA_VERSION
        ):
            raise ValueError(f"Unexpected SFT record schema on line {line_number}")
        task_id = record.get("id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id in seen:
            raise ValueError(f"Invalid/duplicate SFT task id {task_id!r}")
        seen.add(task_id)
        if record.get("gold_action") not in ACTIONS:
            raise ValueError(f"SFT task {task_id} has invalid action")
        if record.get("full_menu_sha256") != expected_full_menu_sha256:
            raise ValueError(f"SFT task {task_id} full-menu hash differs")
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise TypeError(f"SFT task {task_id} messages must be non-empty")
        if record.get("messages_sha256") != canonical_json_sha256(messages):
            raise ValueError(f"SFT task {task_id} messages hash differs")
        input_sha = record.get("input_sha256")
        provenance = record.get("provenance")
        if not isinstance(input_sha, str) or _SHA256_RE.fullmatch(input_sha) is None:
            raise ValueError(f"SFT task {task_id} input hash is invalid")
        if not isinstance(provenance, dict) or provenance.get(
            "input_bundle_sha256"
        ) != manifest.get("input_bundle_sha256"):
            raise ValueError(f"SFT task {task_id} provenance differs from manifest")
        runtime = manifest.get("runtime_provenance")
        if not isinstance(runtime, dict) or provenance.get(
            "runtime_provenance_sha256"
        ) != runtime.get("sha256"):
            raise ValueError(f"SFT task {task_id} runtime provenance differs")
        records.append(record)
    if output.get("n_records") != len(records):
        raise ValueError("SFT manifest record count differs from JSONL")
    if manifest.get("decisions", {}).get("retained") != len(records):
        raise ValueError("SFT decision retained count differs from JSONL")
    decisions = manifest.get("decisions")
    protocol = manifest.get("protocol")
    if not isinstance(decisions, dict) or not isinstance(protocol, dict):
        raise TypeError("SFT manifest lacks protocol/decisions")
    source_count = protocol.get("source_task_count")
    if (
        decisions.get("total") != source_count
        or decisions.get("retained", 0) + decisions.get("dropped", 0) != source_count
    ):
        raise ValueError("SFT source/decision counts do not reconcile")
    task_decisions = decisions.get("task_decisions")
    if not isinstance(task_decisions, list) or len(task_decisions) != source_count:
        raise ValueError("SFT task decision ledger is incomplete")
    decision_ids = [row.get("id") for row in task_decisions if isinstance(row, dict)]
    if len(decision_ids) != source_count or len(set(decision_ids)) != source_count:
        raise ValueError("SFT task decision IDs are malformed or duplicated")
    retained_ids = [
        row.get("id")
        for row in task_decisions
        if isinstance(row, dict) and row.get("decision") == "retained"
    ]
    if retained_ids != [record["id"] for record in records]:
        raise ValueError("SFT retained decision order/IDs differ from JSONL")
    if manifest.get("task_ids_sha256") != canonical_json_sha256(decision_ids):
        raise ValueError("SFT source task ID hash differs from decision ledger")
    present = {record["gold_action"] for record in records}
    if present != set(ACTIONS):
        raise ValueError(
            f"SFT JSONL is missing actions {sorted(set(ACTIONS) - present)}"
        )
    return records, manifest


def _as_token_ids(value: Any, context: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise TypeError(f"{context} token tensor must be rank 1")
        ids = value.tolist()
    elif isinstance(value, (list, tuple)):
        ids = list(value)
    else:
        raise TypeError(f"{context} chat template must return token IDs")
    if not all(isinstance(token, int) and not isinstance(token, bool) for token in ids):
        raise TypeError(f"{context} contains non-integer token IDs")
    return ids


def _render_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    tools: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    value = tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return _as_token_ids(value, "tokenizer.apply_chat_template")


def tokenize_assistant_only(
    record: Mapping[str, Any],
    tokenizer: Any,
    tools: Sequence[dict[str, Any]],
    *,
    max_length: int = MAX_LENGTH,
) -> dict[str, torch.Tensor]:
    """Tokenize one chat and prove each assistant span by prefix difference."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(tools) != EXPECTED_TOOL_COUNT:
        raise ValueError(f"Training chat template requires {EXPECTED_TOOL_COUNT} tools")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TypeError("SFT record messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise TypeError(f"SFT message {index} must contain only role/content")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"SFT message {index} has invalid role")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise ValueError(f"SFT message {index} has empty content")
    full_ids = _render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    task_id = record.get("id", "unknown")
    if len(full_ids) > max_length:
        raise ValueError(
            f"SFT task {task_id} has {len(full_ids)} tokens, exceeding max_length={max_length}; truncation is forbidden"
        )
    labels = [-100] * len(full_ids)
    assistant_spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        after = _render_ids(
            tokenizer, messages[: index + 1], tools, add_generation_prompt=False
        )
        if full_ids[: len(after)] != after:
            raise ValueError(
                f"SFT task {task_id} prefix {index + 1} is not stable in full rendering"
            )
        before = (
            []
            if index == 0
            else _render_ids(
                tokenizer, messages[:index], tools, add_generation_prompt=False
            )
        )
        if after[: len(before)] != before:
            raise ValueError(f"SFT task {task_id} message-prefix tokenization changed")
        if message["role"] != "assistant":
            continue
        generation_prefix = _render_ids(
            tokenizer, messages[:index], tools, add_generation_prompt=True
        )
        if generation_prefix[: len(before)] != before:
            raise ValueError(
                f"SFT task {task_id} assistant generation prefix changed prior messages"
            )
        if after[: len(generation_prefix)] != generation_prefix:
            raise ValueError(
                f"SFT task {task_id} assistant span cannot be proven by prefix difference"
            )
        start, end = len(generation_prefix), len(after)
        if end <= start:
            raise ValueError(f"SFT task {task_id} has an empty assistant token span")
        if assistant_spans and start < assistant_spans[-1][1]:
            raise AssertionError("Assistant label spans overlap")
        labels[start:end] = full_ids[start:end]
        assistant_spans.append((start, end))
    if not assistant_spans or all(label == -100 for label in labels):
        raise ValueError(f"SFT task {task_id} has no trainable assistant tokens")
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class AssistantOnlyDataset(torch.utils.data.Dataset):
    """Eager validation keeps sequence failures out of the training loop."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        tools: Sequence[dict[str, Any]],
        *,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if not records:
            raise ValueError("AssistantOnlyDataset requires at least one record")
        self.features = [
            tokenize_assistant_only(record, tokenizer, tools, max_length=max_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError("pad_token_id must be an integer")
        self.pad_token_id = pad_token_id

    def __call__(
        self, features: Sequence[Mapping[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        width = max(int(feature["input_ids"].numel()) for feature in features)
        batch_size = len(features)
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for row, feature in enumerate(features):
            length = int(feature["input_ids"].numel())
            if (
                feature["attention_mask"].numel() != length
                or feature["labels"].numel() != length
            ):
                raise ValueError("Feature tensor lengths differ")
            input_ids[row, :length] = feature["input_ids"]
            attention_mask[row, :length] = feature["attention_mask"]
            labels[row, :length] = feature["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_offline_tokenizer(model_path: Path | str) -> Any:
    """Load the registered tokenizer without network or remote Python code."""

    source = Path(model_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    if (
        not isinstance(getattr(tokenizer, "chat_template", None), str)
        or not tokenizer.chat_template
    ):
        raise ValueError("Tokenizer has no chat_template")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad token could not be set")
    return tokenizer
