"""Fail-closed semantic audit for the completed formal statistics stage.

The handoff inventory freezes files.  This module performs the complementary
semantic checks that bind behavior, statistics publications, Probe&Prefill,
the scoped relabel derivation, and the imported original-W2T probe before the
handoff inventory is built.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .constants import EXPECTED_SPLIT_SIZES
from .constants import SCHEMA_VERSION as ACTION_SCHEMA_VERSION
from .constants import UPSTREAM_COMMIT
from .evaluation_contract import (
    ActionRowContract,
    classify_action_outcome,
    validate_action_row,
)
from .io_utils import atomic_write_json, canonical_json_sha256
from .scoped_relabel import ALLOWED_ROW_MUTATIONS
from .stage_handoff import FORMAL_BEHAVIOR_OUTPUTS


AUDIT_SCHEMA_VERSION = "when2tool-formal-stage-audit.v1"
AUDIT_MANIFEST_TYPE = "formal-statistics-stage-semantic-audit"
MODEL_SLUG = "qwen3-4b-instruct-2507"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_TRAIN_TASK_COUNT = EXPECTED_SPLIT_SIZES["train"]
EXPECTED_TASK_COUNT = EXPECTED_SPLIT_SIZES["test"]
EXPECTED_MAX_ROUNDS = 10
EXPECTED_PROBE_TEMPERATURE = 2.0
EXPECTED_PROBE_LABEL_SEED = 0
EXPECTED_PROBE_N_LAYERS = 37
EXPECTED_PROBE_HIDDEN_DIM = 2560
EXPECTED_PROBE_C = 0.0001
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
STATISTICS_SCHEMA_VERSION = "when2tool_action_stats.v3"
RELABEL_PROTOCOL = "scoped_original_w2t"
# Backward-compatible public name, sourced from the relabel implementation so
# the producer and final auditor cannot silently grow different allowlists.
RELABEL_ALLOWED_ROW_MUTATIONS = ALLOWED_ROW_MUTATIONS
AUDIT_RECEIPT_RELATIVE = PurePosixPath("manifests/formal_stage_audit.json")

FULL_SETTINGS = (
    "current_no_reasoning_fulltools",
    "necessary_tool_no_reasoning_fulltools",
    "sparse_tool_no_reasoning_fulltools",
    *(f"probe_prefill_t{threshold:.1f}_fulltools" for threshold in THRESHOLDS),
)
SCOPED_PROMPT_SETTINGS = tuple(
    f"{prompt}_{reasoning}_scoped"
    for reasoning in ("no_reasoning", "reasoning")
    for prompt in (
        "force_tool",
        "current",
        "necessary_tool",
        "sparse_tool",
        "no_tool",
    )
)
SCOPED_ADAPTED_SETTINGS = (
    *SCOPED_PROMPT_SETTINGS,
    *(f"probe_prefill_t{threshold:.1f}_scoped" for threshold in THRESHOLDS),
)
SCOPED_ORIGINAL_SETTINGS = (
    *SCOPED_PROMPT_SETTINGS,
    *(
        f"probe_prefill_t{threshold:.1f}_scoped_original_w2t"
        for threshold in THRESHOLDS
    ),
)


@dataclass(frozen=True)
class BehaviorMode:
    prompt_mode: str
    reasoning_mode: str
    record_mode: str = "lite"


def _build_behavior_modes() -> dict[str, BehaviorMode]:
    modes = {
        "current_no_reasoning_fulltools": BehaviorMode("current", "no_reasoning"),
        "necessary_tool_no_reasoning_fulltools": BehaviorMode(
            "necessary_tool", "no_reasoning"
        ),
        "sparse_tool_no_reasoning_fulltools": BehaviorMode(
            "sparse_tool", "no_reasoning"
        ),
    }
    for threshold in THRESHOLDS:
        modes[f"probe_prefill_t{threshold:.1f}_fulltools"] = BehaviorMode(
            "current", "no_reasoning"
        )
        modes[f"probe_prefill_t{threshold:.1f}_scoped"] = BehaviorMode(
            "current", "no_reasoning"
        )
        modes[f"probe_prefill_t{threshold:.1f}_scoped_original_w2t"] = BehaviorMode(
            "current", "no_reasoning"
        )
    for reasoning in ("no_reasoning", "reasoning"):
        for prompt in (
            "force_tool",
            "current",
            "necessary_tool",
            "sparse_tool",
            "no_tool",
        ):
            modes[f"{prompt}_{reasoning}_scoped"] = BehaviorMode(prompt, reasoning)
    expected = {
        *FULL_SETTINGS,
        *SCOPED_ADAPTED_SETTINGS,
        *SCOPED_ORIGINAL_SETTINGS,
    }
    if set(modes) != expected:
        raise AssertionError("Behavior-mode registry differs from formal settings")
    return modes


BEHAVIOR_MODES = _build_behavior_modes()

PUBLISHED_ANALYSIS_FILES = tuple(
    sorted(
        (
            "action_recall_summary.csv",
            "accuracy_vs_total_tc.png",
            "accuracy_vs_total_tc_by_difficulty.png",
            "class_outcome_rates.csv",
            "confusion_counts.csv",
            "confusion_heatmap.png",
            "confusion_row_normalized.csv",
            "current_relative_tradeoff_per_run.csv",
            "current_relative_tradeoff_summary.csv",
            "derived_action_rows.csv",
            "difficulty_current_relative_tradeoff_per_run.csv",
            "difficulty_current_relative_tradeoff_summary.csv",
            "difficulty_metric_summary.csv",
            "difficulty_per_run_metrics.csv",
            "error_stacked.png",
            "gold_action_final_accuracy_per_run.csv",
            "gold_action_final_accuracy_summary.csv",
            "label_distribution.csv",
            "multicall_by_gold.csv",
            "multicall_by_gold_summary.csv",
            "multicall_metric_summary.csv",
            "multicall_summary.csv",
            "needed_category_analysis.csv",
            "needed_category_analysis_summary.csv",
            "none_analysis.csv",
            "none_analysis_summary.csv",
            "outcome_counts.csv",
            "outcome_rates.csv",
            "paired_bootstrap_comparisons.csv",
            "per_run_metrics.csv",
            "recall_bars.png",
            "run_diagnostic_summary.csv",
            "run_diagnostics.csv",
            "setting_metric_summary.csv",
        )
    )
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class FormalProtocol:
    """Fixed production protocol with a small test-only task-count seam."""

    model_slug: str = MODEL_SLUG
    seeds: tuple[int, ...] = EXPECTED_SEEDS
    train_task_count: int = EXPECTED_TRAIN_TASK_COUNT
    task_count: int = EXPECTED_TASK_COUNT
    max_rounds: int = EXPECTED_MAX_ROUNDS
    probe_temperature: float = EXPECTED_PROBE_TEMPERATURE
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 20260722


FORMAL_PROTOCOL = FormalProtocol()


@dataclass(frozen=True)
class MigrationValidationContext:
    root: Path
    model_slug: str
    config_sha256: str
    task_ids: tuple[int, ...]
    labels: Mapping[str, Any]
    label_seed: int = EXPECTED_PROBE_LABEL_SEED
    n_layers: int = EXPECTED_PROBE_N_LAYERS
    hidden_dim: int = EXPECTED_PROBE_HIDDEN_DIM
    probe_c: float = EXPECTED_PROBE_C


MigrationValidator = Callable[[Mapping[str, Any], MigrationValidationContext], None]
ScopedMenuBuilder = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class AnalysisSpec:
    protocol: str
    output_directory: str
    settings: tuple[str, ...]
    label_protocol: str
    tool_scope: str
    label_stem: str
    data_filename: str

    def expected_rows(self, formal: FormalProtocol) -> int:
        return len(self.settings) * len(formal.seeds) * formal.task_count

    def expected_runs(self, formal: FormalProtocol) -> int:
        return len(self.settings) * len(formal.seeds)


ANALYSIS_SPECS = (
    AnalysisSpec(
        protocol="fulltools",
        output_directory="fulltools",
        settings=FULL_SETTINGS,
        label_protocol="adapted",
        tool_scope="full",
        label_stem="fulltools",
        data_filename="tasks_v1_test_fulltools_category.json",
    ),
    AnalysisSpec(
        protocol="scoped_adapted",
        output_directory="scoped_adapted",
        settings=SCOPED_ADAPTED_SETTINGS,
        label_protocol="adapted",
        tool_scope="scoped",
        label_stem="scoped",
        data_filename="tasks_v1_test_category.json",
    ),
    AnalysisSpec(
        protocol="scoped_original_w2t",
        output_directory="scoped_original_w2t",
        settings=SCOPED_ORIGINAL_SETTINGS,
        label_protocol=RELABEL_PROTOCOL,
        tool_scope="scoped",
        label_stem=RELABEL_PROTOCOL,
        data_filename="tasks_v1_test_category.json",
    ),
)


@dataclass(frozen=True)
class FileSnapshot:
    path_base: str
    path: str
    absolute: Path
    bytes: int
    sha256: str
    signature: tuple[int, int, int, int]


class SnapshotRegistry:
    """Hash inputs once and verify their filesystem identities before publish."""

    def __init__(self, run_root: Path, repository_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.repository_root = repository_root.resolve()
        self._items: dict[tuple[str, str], FileSnapshot] = {}

    @staticmethod
    def _signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )

    def _relative(self, path: Path, path_base: str) -> str:
        base = self.run_root if path_base == "run_root" else self.repository_root
        try:
            relative = path.relative_to(base)
        except ValueError as error:
            raise ValueError(f"Audited file escapes {path_base}: {path}") from error
        value = relative.as_posix()
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or any(
            part in {"", ".", ".."} for part in parsed.parts
        ):
            raise ValueError(f"Non-canonical audited path: {value!r}")
        return value

    def snapshot(self, path: Path, *, path_base: str = "run_root") -> FileSnapshot:
        absolute = path.resolve()
        relative = self._relative(absolute, path_base)
        key = (path_base, relative)
        if key in self._items:
            return self._items[key]
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"Audited input must be a regular non-symlink file: {path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if self._signature(opened_before) != self._signature(before):
                raise RuntimeError(f"Audited input changed while opening: {path}")
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
        signature = self._signature(after)
        if (
            self._signature(before) != self._signature(opened_after)
            or self._signature(before) != signature
        ):
            raise RuntimeError(f"Audited input changed while hashing: {path}")
        item = FileSnapshot(
            path_base=path_base,
            path=relative,
            absolute=absolute,
            bytes=after.st_size,
            sha256=digest.hexdigest(),
            signature=signature,
        )
        self._items[key] = item
        return item

    def read_bytes(
        self, path: Path, *, path_base: str = "run_root"
    ) -> tuple[bytes, FileSnapshot]:
        """Read one file once and bind the exact returned bytes to its snapshot."""

        absolute = path.resolve()
        relative = self._relative(absolute, path_base)
        key = (path_base, relative)
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"Audited input must be a regular non-symlink file: {path}"
            )
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if self._signature(opened_before) != self._signature(before):
                raise RuntimeError(f"Audited input changed while opening: {path}")
            payload = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = path.lstat()
        signature = self._signature(after)
        if (
            self._signature(before) != self._signature(opened_after)
            or self._signature(before) != signature
        ):
            raise RuntimeError(f"Audited input changed while reading: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        existing = self._items.get(key)
        if existing is not None:
            if existing.signature != signature or existing.sha256 != digest:
                raise RuntimeError(f"Audited input changed between reads: {path}")
            return payload, existing
        item = FileSnapshot(
            path_base=path_base,
            path=relative,
            absolute=absolute,
            bytes=after.st_size,
            sha256=digest,
            signature=signature,
        )
        self._items[key] = item
        return payload, item

    def register_verified(
        self,
        path: Path,
        expected_sha256: str,
        *,
        path_base: str = "run_root",
    ) -> FileSnapshot:
        expected_sha256 = _require_sha256(expected_sha256, f"verified SHA for {path}")
        item = self.snapshot(path, path_base=path_base)
        if item.sha256 != expected_sha256:
            raise ValueError(
                f"Verified artifact SHA256 mismatch for {path}: "
                f"{item.sha256} != {expected_sha256}"
            )
        return item

    def ensure_unchanged(self) -> None:
        for item in self._items.values():
            metadata = item.absolute.lstat()
            if self._signature(metadata) != item.signature:
                raise RuntimeError(
                    f"Audited input changed after validation: {item.absolute}"
                )

    def receipt_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "path_base": item.path_base,
                "path": item.path,
                "bytes": item.bytes,
                "sha256": item.sha256,
            }
            for item in sorted(
                self._items.values(), key=lambda value: (value.path_base, value.path)
            )
        ]


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON list")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _require_binary_int(value: Any, context: str) -> int:
    result = _require_int(value, context)
    if result not in (0, 1):
        raise ValueError(f"{context} must be integer 0 or 1")
    return result


def _require_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def _require_commit(value: Any, context: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical lowercase 40-hex Git commit")
    return value


def _expect(mapping: Mapping[str, Any], key: str, expected: Any, context: str) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{context}.{key}={actual!r}, expected {expected!r}")


def _read_json(
    path: Path,
    registry: SnapshotRegistry,
    context: str,
    *,
    path_base: str = "run_root",
) -> tuple[dict[str, Any], FileSnapshot]:
    payload, item = registry.read_bytes(path, path_base=path_base)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {context}: {path}") from error
    return dict(_require_mapping(value, context)), item


def _resolved_run_path(root: Path, relative: str, context: str) -> Path:
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(
            f"{context} is not a canonical run-root relative path: {relative!r}"
        )
    target = root.joinpath(*parsed.parts).resolve()
    if root not in target.parents:
        raise ValueError(f"{context} escapes run root: {relative!r}")
    return target


def _validate_exact_output_inventory(root: Path) -> None:
    output_root = root / "outputs"
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("outputs must be a real directory")
    files: set[str] = set()
    directories: set[str] = set()
    for current, directory_names, file_names in os.walk(
        output_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError(f"Output directory must be real: {relative}")
            directories.add(relative)
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(f"Output artifact must be regular: {relative}")
            files.add(relative)
    expected_directories = {
        "outputs/fulltools",
        "outputs/scoped_adapted",
        "outputs/scoped_original_w2t",
    }
    expected_files = {
        *FORMAL_BEHAVIOR_OUTPUTS,
        "outputs/scoped_original_w2t/relabel_receipt.json",
    }
    if directories != expected_directories:
        raise ValueError(
            "Formal output directory inventory mismatch: "
            f"missing={sorted(expected_directories - directories)}, "
            f"extra={sorted(directories - expected_directories)}"
        )
    if files != expected_files:
        raise ValueError(
            "Formal output file inventory mismatch: "
            f"missing={sorted(expected_files - files)}, extra={sorted(files - expected_files)}"
        )


def _expected_behavior_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in ANALYSIS_SPECS:
        for setting in spec.settings:
            path = root / "outputs" / spec.output_directory / f"{setting}.json"
            key = path.relative_to(root).as_posix()
            if key in paths:
                raise AssertionError(f"Duplicate registered behavior path: {key}")
            paths[key] = path
    if set(paths) != set(FORMAL_BEHAVIOR_OUTPUTS):
        raise AssertionError("Audit behavior registry differs from handoff registry")
    return paths


def _validate_runtime_and_config(
    root: Path,
    config_path: Path,
    repository_root: Path,
    registry: SnapshotRegistry,
    *,
    behavior_commit: str,
    formal: FormalProtocol,
) -> tuple[dict[str, Any], FileSnapshot, FileSnapshot]:
    provenance_path = root / "manifests/runtime_provenance.json"
    provenance, provenance_file = _read_json(
        provenance_path, registry, "runtime provenance"
    )
    _expect(provenance, "manifest_type", "runtime-and-input-provenance", "provenance")
    _expect(provenance, "schema_version", ACTION_SCHEMA_VERSION, "provenance")
    _expect(provenance, "upstream_commit", UPSTREAM_COMMIT, "provenance")
    git = _require_mapping(provenance.get("git"), "provenance.git")
    _expect(git, "commit", behavior_commit, "provenance.git")
    _expect(git, "worktree_clean", True, "provenance.git")
    registered_config = _require_mapping(provenance.get("config"), "provenance.config")
    _expect(registered_config, "model_slug", formal.model_slug, "provenance.config")
    registered_relative = _require_string(
        registered_config.get("path"), "provenance.config.path"
    )
    registered_path = _resolved_repository_path(
        repository_root, registered_relative, "provenance.config.path"
    )
    if config_path.resolve() != registered_path:
        raise ValueError(
            f"Explicit config {config_path.resolve()} differs from registered config {registered_path}"
        )
    config_file = registry.snapshot(config_path, path_base="code_repository")
    expected_config_sha = _require_sha256(
        registered_config.get("sha256"), "provenance.config.sha256"
    )
    if config_file.sha256 != expected_config_sha:
        raise ValueError(
            "Registered config SHA256 differs from the explicit config file"
        )
    return provenance, provenance_file, config_file


def _resolved_repository_path(root: Path, relative: str, context: str) -> Path:
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{context} is not a canonical repository-relative path")
    target = root.joinpath(*parsed.parts).resolve()
    if root not in target.parents:
        raise ValueError(f"{context} escapes the code repository")
    return target


def _is_derived_scoped_prompt(relative: str) -> bool:
    return relative.startswith("outputs/scoped_original_w2t/") and Path(
        relative
    ).stem in {*SCOPED_PROMPT_SETTINGS}


def _behavior_spec_for_relative(relative: str) -> AnalysisSpec:
    for spec in ANALYSIS_SPECS:
        prefix = f"outputs/{spec.output_directory}/"
        if relative.startswith(prefix):
            return spec
    raise AssertionError(relative)


def _validated_label_rows_by_id(
    payload: Mapping[str, Any],
    context: str,
    formal: FormalProtocol,
    *,
    split: str,
    tool_scope: str,
    protocol_id: str | None,
) -> tuple[tuple[int, ...], dict[int, Mapping[str, Any]]]:
    for key, expected in {
        "schema_version": ACTION_SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": formal.model_slug,
        "split": split,
        "seed": EXPECTED_PROBE_LABEL_SEED,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": tool_scope,
    }.items():
        _expect(payload, key, expected, context)
    if protocol_id is None:
        if payload.get("protocol_id") is not None:
            raise ValueError(f"{context}.protocol_id must be absent for adapted labels")
    else:
        _expect(payload, "protocol_id", protocol_id, context)
    rows = _require_list(payload.get("rows"), f"{context}.rows")
    expected_count = formal.train_task_count if split == "train" else formal.task_count
    if len(rows) != expected_count:
        raise ValueError(
            f"{context}: expected {expected_count} {split} label rows, got {len(rows)}"
        )
    _expect(payload, "n", len(rows), context)
    if not rows:
        raise ValueError(f"{context}.rows must be non-empty")

    order: list[int] = []
    by_id: dict[int, Mapping[str, Any]] = {}
    required = {
        "id",
        "split",
        "difficulty",
        "category",
        "tool_necessary",
        "no_tool_correct",
        "gold_action",
        "seed",
        "prompt_mode",
        "reasoning_mode",
        "tool_scope",
    }
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"{context}.rows[{index}]")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{context}.rows[{index}] is missing {missing}")
        task_id = _require_int(row.get("id"), f"{context}.rows[{index}].id")
        if task_id in by_id:
            raise ValueError(f"{context}: duplicate label ID {task_id}")
        for key, expected in {
            "split": split,
            "seed": EXPECTED_PROBE_LABEL_SEED,
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": tool_scope,
        }.items():
            _expect(row, key, expected, f"{context}.rows[{index}]")
        category = row.get("category")
        if category not in {"A", "B", "C"}:
            raise ValueError(f"{context}.rows[{index}].category is invalid")
        _require_string(row.get("difficulty"), f"{context}.rows[{index}].difficulty")
        necessary = _require_binary_int(
            row.get("tool_necessary"),
            f"{context}.rows[{index}].tool_necessary",
        )
        no_tool_correct = _require_binary_int(
            row.get("no_tool_correct"),
            f"{context}.rows[{index}].no_tool_correct",
        )
        if no_tool_correct != 1 - necessary:
            raise ValueError(
                f"{context}.rows[{index}].no_tool_correct disagrees with tool_necessary"
            )
        _expect(
            row,
            "gold_action",
            category if necessary else "NONE",
            f"{context}.rows[{index}]",
        )
        if protocol_id is not None:
            _expect(
                row,
                "label_protocol",
                protocol_id,
                f"{context}.rows[{index}]",
            )
        order.append(task_id)
        by_id[task_id] = row
    return tuple(order), by_id


def _validate_behavior_row_against_label(
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    context: str,
) -> ActionRowContract:
    for key in ("category", "difficulty", "gold_action"):
        _expect(row, key, label.get(key), context)
    # These fields are label-derived and not guaranteed to be copied into the
    # runtime row.  If present, they must be exact; the label artifact itself
    # is always required to carry and internally validate both fields.
    for key in ("tool_necessary", "no_tool_correct"):
        if key in row:
            _expect(row, key, label.get(key), context)
    if "error_type" not in row:
        raise ValueError(f"{context} is missing required action field error_type")
    return validate_action_row(
        row,
        context=context,
        require_pred_action=True,
        strict_event_categories=True,
        require_formal_diagnostics=True,
    )


def _default_scoped_menu_builder(task: Mapping[str, Any]) -> str:
    # The pinned upstream runtime imports NumPy-heavy environments, so keep it
    # lazy while ensuring every production/CLI audit uses the evaluation path.
    from .upstream import build_environments

    return build_environments(dict(task), "scoped").menu_sha256


def _rebuild_scoped_menu_by_id(
    root: Path,
    registry: SnapshotRegistry,
    formal: FormalProtocol,
    menu_builder: ScopedMenuBuilder,
) -> tuple[tuple[int, ...], dict[int, str]]:
    """Rebuild each scoped schema menu through the pinned production adapter."""

    data_path = root / "data/tasks_v1_test_category.json"
    payload, _ = registry.read_bytes(data_path)
    try:
        tasks = _require_list(json.loads(payload), "pinned scoped test task data")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid UTF-8 JSON in scoped task data: {data_path}"
        ) from error
    if len(tasks) != formal.task_count:
        raise ValueError(
            f"Scoped test task data has {len(tasks)} rows, expected {formal.task_count}"
        )
    order: list[int] = []
    menus: dict[int, str] = {}
    for index, raw_task in enumerate(tasks):
        task = _require_mapping(raw_task, f"scoped task data[{index}]")
        task_id = _require_int(task.get("id"), f"scoped task data[{index}].id")
        if task_id in menus:
            raise ValueError(f"Scoped task data contains duplicate ID {task_id}")
        _expect(task, "tool_scope", "scoped", f"scoped task data[{index}]")
        menu_sha256 = _require_sha256(
            menu_builder(task),
            f"scoped task {task_id} rebuilt menu SHA256",
        )
        order.append(task_id)
        menus[task_id] = menu_sha256
    return tuple(order), menus


def _validate_behavior_artifacts(
    root: Path,
    registry: SnapshotRegistry,
    *,
    provenance_sha256: str,
    config_sha256: str,
    full_menu_sha256: str,
    behavior_commit: str,
    formal: FormalProtocol,
    scoped_menu_builder: ScopedMenuBuilder,
) -> tuple[dict[str, FileSnapshot], tuple[int, ...]]:
    snapshots: dict[str, FileSnapshot] = {}
    reference_ids: tuple[int, ...] | None = None
    label_cache: dict[str, tuple[tuple[int, ...], dict[int, Mapping[str, Any]]]] = {}
    scoped_task_order, scoped_menu_by_id = _rebuild_scoped_menu_by_id(
        root, registry, formal, scoped_menu_builder
    )
    for relative, path in sorted(_expected_behavior_paths(root).items()):
        payload, item = _read_json(path, registry, f"behavior artifact {relative}")
        snapshots[relative] = item
        derived = _is_derived_scoped_prompt(relative)
        expected_top = {"schema_version", "upstream_commit", "config", "runs"}
        if derived:
            expected_top.add("derivation")
        if set(payload) != expected_top:
            raise ValueError(
                f"{relative}: top-level keys {sorted(payload)} != {sorted(expected_top)}"
            )
        _expect(payload, "schema_version", ACTION_SCHEMA_VERSION, relative)
        _expect(payload, "upstream_commit", UPSTREAM_COMMIT, relative)
        config = _require_mapping(payload.get("config"), f"{relative}.config")
        setting = path.stem
        spec = _behavior_spec_for_relative(relative)
        mode = BEHAVIOR_MODES[setting]
        for key, expected in {
            "model": formal.model_slug,
            "setting": setting,
            "tool_scope": spec.tool_scope,
            "prompt_mode": mode.prompt_mode,
            "reasoning_mode": mode.reasoning_mode,
            "record_mode": mode.record_mode,
            "seeds": list(formal.seeds),
            "smoke": False,
            "max_rounds": formal.max_rounds,
            "project_git_commit": behavior_commit,
            "runtime_provenance_sha256": provenance_sha256,
            "config_sha256": config_sha256,
            "full_menu_sha256": full_menu_sha256,
        }.items():
            _expect(config, key, expected, f"{relative}.config")
        task_ids_sha256 = _require_sha256(
            config.get("task_ids_sha256"), f"{relative}.config.task_ids_sha256"
        )
        _require_sha256(config.get("data_sha256"), f"{relative}.config.data_sha256")
        _require_sha256(config.get("labels_sha256"), f"{relative}.config.labels_sha256")
        data_file = registry.snapshot(root / "data" / spec.data_filename)
        _expect(config, "data_sha256", data_file.sha256, f"{relative}.config")
        label_file = registry.snapshot(
            root
            / "labels"
            / formal.model_slug
            / f"test_labels_no_reasoning_{spec.label_stem}.json"
        )
        _expect(config, "labels_sha256", label_file.sha256, f"{relative}.config")
        if spec.label_stem not in label_cache:
            label_payload, _ = _read_json(
                label_file.absolute,
                registry,
                f"{spec.protocol} test labels",
            )
            label_cache[spec.label_stem] = _validated_label_rows_by_id(
                label_payload,
                f"{spec.protocol} test labels",
                formal,
                split="test",
                tool_scope=spec.tool_scope,
                protocol_id=(
                    RELABEL_PROTOCOL
                    if spec.label_protocol == RELABEL_PROTOCOL
                    else None
                ),
            )
        label_order, labels_by_id = label_cache[spec.label_stem]
        runs = _require_list(payload.get("runs"), f"{relative}.runs")
        if len(runs) != len(formal.seeds):
            raise ValueError(
                f"{relative}: expected {len(formal.seeds)} runs, got {len(runs)}"
            )
        artifact_ids: tuple[int, ...] | None = None
        for run_index, raw_run in enumerate(runs):
            seed = formal.seeds[run_index]
            run = _require_mapping(raw_run, f"{relative}.runs[{run_index}]")
            run_id = f"run_{run_index}_seed_{seed}"
            for key, expected in {
                "run_id": run_id,
                "seed": seed,
                "setting": setting,
            }.items():
                _expect(run, key, expected, f"{relative}.runs[{run_index}]")
            rows = _require_list(run.get("rows"), f"{relative}.runs[{run_index}].rows")
            if len(rows) != formal.task_count:
                raise ValueError(
                    f"{relative} run {run_id}: expected {formal.task_count} rows, "
                    f"got {len(rows)}"
                )
            ids: list[int] = []
            for row_index, raw_row in enumerate(rows):
                row = _require_mapping(
                    raw_row, f"{relative}.runs[{run_index}].rows[{row_index}]"
                )
                task_id = _require_int(
                    row.get("id"), f"{relative}.runs[{run_index}].rows[{row_index}].id"
                )
                ids.append(task_id)
                for key, expected in {
                    "schema_version": ACTION_SCHEMA_VERSION,
                    "run_id": run_id,
                    "seed": seed,
                    "setting": setting,
                    "tool_scope": spec.tool_scope,
                }.items():
                    _expect(
                        row,
                        key,
                        expected,
                        f"{relative}.runs[{run_index}].rows[{row_index}]",
                    )
                context = f"{relative}.runs[{run_index}].rows[{row_index}]"
                if task_id not in labels_by_id:
                    raise ValueError(f"{context}: task ID is absent from formal labels")
                _validate_behavior_row_against_label(
                    row, labels_by_id[task_id], context
                )
                menu_sha256 = _require_sha256(
                    row.get("menu_sha256"), f"{context}.menu_sha256"
                )
                if spec.tool_scope == "full":
                    if menu_sha256 != full_menu_sha256:
                        raise ValueError(
                            f"{context}.menu_sha256 differs from provenance full menu"
                        )
                else:
                    expected_scoped_menu = scoped_menu_by_id.get(task_id)
                    if expected_scoped_menu is None:
                        raise ValueError(
                            f"{context}: task ID is absent from pinned scoped task data"
                        )
                    if menu_sha256 != expected_scoped_menu:
                        raise ValueError(
                            f"{context}.menu_sha256 differs from the menu rebuilt "
                            "from pinned scoped task data"
                        )
            if len(ids) != len(set(ids)):
                raise ValueError(f"{relative} run {run_id} contains duplicate task IDs")
            ids_tuple = tuple(ids)
            if artifact_ids is None:
                artifact_ids = ids_tuple
            elif artifact_ids != ids_tuple:
                raise ValueError(f"{relative}: task ID order varies across seeds")
        assert artifact_ids is not None
        if artifact_ids != label_order:
            raise ValueError(
                f"{relative}: behavior task ID order differs from formal test labels"
            )
        if spec.tool_scope == "scoped" and artifact_ids != scoped_task_order:
            raise ValueError(
                f"{relative}: behavior task ID order differs from scoped task data"
            )
        if canonical_json_sha256(list(artifact_ids)) != task_ids_sha256:
            raise ValueError(f"{relative}: task_ids_sha256 disagrees with row IDs")
        if reference_ids is None:
            reference_ids = artifact_ids
        elif reference_ids != artifact_ids:
            raise ValueError(
                f"{relative}: task ID order differs across formal artifacts"
            )
    assert reference_ids is not None
    return snapshots, reference_ids


def _analysis_behavior_paths(root: Path, spec: AnalysisSpec) -> tuple[Path, ...]:
    return tuple(
        root / "outputs" / spec.output_directory / f"{setting}.json"
        for setting in spec.settings
    )


def _validate_analysis_directory(
    directory: Path,
    published: Mapping[str, str],
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Analysis output must be a real directory: {directory}")
    entries = list(directory.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError(f"Analysis directory contains a symlink: {directory}")
    if any(not path.is_file() for path in entries):
        raise ValueError(f"Analysis directory contains a non-file entry: {directory}")
    actual = {path.name for path in entries}
    expected = {"summary.json", *PUBLISHED_ANALYSIS_FILES}
    if actual != expected:
        raise ValueError(
            f"Analysis file inventory mismatch for {directory.name}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if set(published) != set(PUBLISHED_ANALYSIS_FILES):
        raise ValueError(f"summary published_files mismatch for {directory.name}")


def _validate_summary_hash_list(
    values: Any,
    context: str,
    *,
    expected_paths: Sequence[Path],
    registry: SnapshotRegistry,
) -> dict[Path, dict[str, Any]]:
    rows = _require_list(values, context)
    if len(rows) != len(expected_paths):
        raise ValueError(
            f"{context}: expected {len(expected_paths)} entries, got {len(rows)}"
        )
    expected_by_path = {path.resolve(): path for path in expected_paths}
    if len(expected_by_path) != len(expected_paths):
        raise AssertionError(f"{context}: duplicate expected paths")
    result: dict[Path, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"{context}[{index}]")
        actual_path = Path(
            _require_string(row.get("path"), f"{context}[{index}].path")
        ).resolve()
        if actual_path not in expected_by_path:
            raise ValueError(f"{context}[{index}].path is outside the registered panel")
        if actual_path in result:
            raise ValueError(f"{context} contains duplicate path {actual_path}")
        expected_path = expected_by_path[actual_path]
        expected_hash = _require_sha256(row.get("sha256"), f"{context}[{index}].sha256")
        registry.register_verified(expected_path, expected_hash)
        result[actual_path] = dict(row)
    if set(result) != set(expected_by_path):
        raise ValueError(f"{context} does not cover the exact registered panel")
    return result


def _validate_analysis_summaries(
    root: Path,
    registry: SnapshotRegistry,
    behavior_snapshots: Mapping[str, FileSnapshot],
    *,
    runtime_provenance_sha256: str,
    behavior_commit: str,
    statistics_commit: str,
    formal: FormalProtocol,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for spec in ANALYSIS_SPECS:
        directory = root / "analysis" / spec.protocol
        summary_path = directory / "summary.json"
        summary, _ = _read_json(summary_path, registry, f"{spec.protocol} summary")
        summaries[spec.protocol] = summary
        for key, expected in {
            "schema_version": STATISTICS_SCHEMA_VERSION,
            "manifest_type": "action-statistics-publication-receipt",
            "publication_complete": True,
            "analysis_protocol": spec.protocol,
            "label_protocol": spec.label_protocol,
            "behavior_generation_git_commit": behavior_commit,
            "statistics_code_git_commit": statistics_commit,
            "expected_seeds": list(formal.seeds),
            "n_rows": spec.expected_rows(formal),
            "n_task_ids": formal.task_count,
            "n_settings": len(spec.settings),
            "n_runs": spec.expected_runs(formal),
        }.items():
            _expect(summary, key, expected, f"analysis/{spec.protocol}/summary.json")
        raw_expected_settings = _require_list(
            summary.get("expected_settings"),
            f"{spec.protocol}.expected_settings",
        )
        expected_settings = [
            _require_string(value, f"{spec.protocol}.expected_settings[{index}]")
            for index, value in enumerate(raw_expected_settings)
        ]
        if len(expected_settings) != len(set(expected_settings)) or set(
            expected_settings
        ) != set(spec.settings):
            raise ValueError(
                f"{spec.protocol}: expected_settings must be the exact registered set"
            )
        paired = _require_mapping(
            summary.get("paired_bootstrap"), f"{spec.protocol}.paired_bootstrap"
        )
        _expect(
            paired,
            "n_bootstrap",
            formal.bootstrap_samples,
            f"{spec.protocol}.paired_bootstrap",
        )
        _expect(
            paired, "seed", formal.bootstrap_seed, f"{spec.protocol}.paired_bootstrap"
        )
        settings = _require_mapping(
            summary.get("settings"), f"{spec.protocol}.settings"
        )
        if set(settings) != set(spec.settings):
            raise ValueError(f"{spec.protocol}: summary settings map is incomplete")
        for setting in spec.settings:
            setting_row = _require_mapping(
                settings[setting], f"{spec.protocol}.settings.{setting}"
            )
            _expect(
                setting_row,
                "n_runs",
                len(formal.seeds),
                f"{spec.protocol}.settings.{setting}",
            )

        expected_behavior_paths = _analysis_behavior_paths(root, spec)
        input_artifacts = _require_list(
            summary.get("input_artifacts"), f"{spec.protocol}.input_artifacts"
        )
        if len(input_artifacts) != len(expected_behavior_paths):
            raise ValueError(f"{spec.protocol}: wrong input_artifact count")
        expected_by_setting = {
            setting: root / "outputs" / spec.output_directory / f"{setting}.json"
            for setting in spec.settings
        }
        artifacts_by_setting: dict[str, Mapping[str, Any]] = {}
        for index, raw in enumerate(input_artifacts):
            row = _require_mapping(raw, f"{spec.protocol}.input_artifacts[{index}]")
            setting = _require_string(
                row.get("setting"),
                f"{spec.protocol}.input_artifacts[{index}].setting",
            )
            if setting not in expected_by_setting or setting in artifacts_by_setting:
                raise ValueError(
                    f"{spec.protocol}: unknown or duplicate input setting {setting!r}"
                )
            expected_path = expected_by_setting[setting]
            actual_path = Path(
                _require_string(
                    row.get("path"), f"{spec.protocol}.input_artifacts[{index}].path"
                )
            )
            if actual_path.resolve() != expected_path.resolve():
                raise ValueError(
                    f"{spec.protocol}: behavior input setting/path mismatch"
                )
            relative = expected_path.relative_to(root).as_posix()
            expected_snapshot = behavior_snapshots[relative]
            for key, expected in {
                "sha256": expected_snapshot.sha256,
                "setting": setting,
                "model": formal.model_slug,
                "tool_scope": spec.tool_scope,
                "runtime_provenance_sha256": runtime_provenance_sha256,
                "project_git_commit": behavior_commit,
            }.items():
                _expect(row, key, expected, f"{spec.protocol}.input_artifacts[{index}]")
            artifacts_by_setting[setting] = row
        if set(artifacts_by_setting) != set(expected_by_setting):
            raise ValueError(
                f"{spec.protocol}: input artifact setting panel is incomplete"
            )
        input_files = _require_list(
            summary.get("input_files"), f"{spec.protocol}.input_files"
        )
        resolved_input_files = [Path(str(value)).resolve() for value in input_files]
        if len(resolved_input_files) != len(set(resolved_input_files)):
            raise ValueError(f"{spec.protocol}: input_files contains duplicates")
        # input_files is treated as an unordered inventory.  Setting-to-file
        # semantics come exclusively from the keyed input_artifacts mapping.
        if set(resolved_input_files) != {
            path.resolve() for path in expected_behavior_paths
        }:
            raise ValueError(
                f"{spec.protocol}: input_files differs from the registered panel"
            )

        label_directory = root / "labels" / formal.model_slug
        expected_labels = tuple(
            label_directory / f"{split}_labels_no_reasoning_{spec.label_stem}.json"
            for split in ("train", "test")
        )
        labels_by_path = _validate_summary_hash_list(
            summary.get("labels_files"),
            f"{spec.protocol}.labels_files",
            expected_paths=expected_labels,
            registry=registry,
        )
        if summary.get("labels_file") is not None:
            raise ValueError(
                f"{spec.protocol}: labels_file must be null for train/test panel"
            )
        for split in ("train", "test"):
            path = label_directory / (
                f"{split}_labels_no_reasoning_{spec.label_stem}.json"
            )
            row = labels_by_path[path.resolve()]
            label_payload, _ = _read_json(
                path, registry, f"{spec.protocol} {split} labels"
            )
            _validated_label_rows_by_id(
                label_payload,
                f"{spec.protocol} {split} labels",
                formal,
                split=split,
                tool_scope=spec.tool_scope,
                protocol_id=(
                    RELABEL_PROTOCOL
                    if spec.label_protocol == RELABEL_PROTOCOL
                    else None
                ),
            )
            if split == "test":
                for artifact in artifacts_by_setting.values():
                    _expect(
                        _require_mapping(artifact, "input artifact"),
                        "labels_sha256",
                        row["sha256"],
                        f"{spec.protocol} input artifact",
                    )

        references = _require_mapping(
            summary.get("referenced_inputs"), f"{spec.protocol}.referenced_inputs"
        )
        if set(references) != {"data", "runtime_provenance"}:
            raise ValueError(f"{spec.protocol}: unexpected referenced_inputs keys")
        data_path = root / "data" / spec.data_filename
        for key, expected_path in {
            "data": data_path,
            "runtime_provenance": root / "manifests/runtime_provenance.json",
        }.items():
            row = _require_mapping(
                references[key], f"{spec.protocol}.referenced_inputs.{key}"
            )
            actual_path = Path(
                _require_string(row.get("path"), f"{spec.protocol}.{key}.path")
            )
            if actual_path.resolve() != expected_path.resolve():
                raise ValueError(f"{spec.protocol}: referenced {key} path mismatch")
            registry.register_verified(
                expected_path,
                _require_sha256(row.get("sha256"), f"{spec.protocol}.{key}.sha256"),
            )

        raw_published = _require_list(
            summary.get("published_files"), f"{spec.protocol}.published_files"
        )
        if len(raw_published) != len(PUBLISHED_ANALYSIS_FILES):
            raise ValueError(f"{spec.protocol}: wrong published_files count")
        published: dict[str, str] = {}
        for index, raw in enumerate(raw_published):
            row = _require_mapping(raw, f"{spec.protocol}.published_files[{index}]")
            name = _require_string(
                row.get("path"), f"{spec.protocol}.published_files[{index}].path"
            )
            if PurePosixPath(name).name != name or name in published:
                raise ValueError(
                    f"{spec.protocol}: non-canonical or duplicate published path {name!r}"
                )
            digest = _require_sha256(
                row.get("sha256"), f"{spec.protocol}.published_files[{index}].sha256"
            )
            published[name] = digest
        _validate_analysis_directory(directory, published)
        for name, digest in published.items():
            registry.register_verified(directory / name, digest)
    return summaries


def _probe_fields(
    row: Mapping[str, Any], context: str
) -> tuple[float, float, float, str, str]:
    logit = _require_number(row.get("probe_logit"), f"{context}.probe_logit")
    probability = _require_number(
        row.get("probe_probability"), f"{context}.probe_probability"
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{context}.probe_probability must be in [0,1]")
    temperature = _require_number(
        row.get("probe_temperature"), f"{context}.probe_temperature"
    )
    decision = _require_string(row.get("probe_decision"), f"{context}.probe_decision")
    prefill = _require_string(row.get("probe_prefill"), f"{context}.probe_prefill")
    return logit, probability, temperature, decision, prefill


def _sigmoid_probability(logit: float, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("Probe temperature must be positive")
    scaled = max(-80.0, min(80.0, logit / temperature))
    return 1.0 / (1.0 + math.exp(-scaled))


def _validate_probe_prefill_family(
    root: Path,
    registry: SnapshotRegistry,
    *,
    directory: str,
    suffix: str,
    probe_directory: str,
    probe_protocol: str,
    probe_scope: str,
    formal: FormalProtocol,
) -> dict[str, Any]:
    expected_probe_input_names = {
        "probe_no_reasoning.pt",
        "test_hidden_no_reasoning.pt",
        "test_labels_no_reasoning.json",
    }
    if probe_protocol == RELABEL_PROTOCOL:
        expected_probe_input_names.add("migration_receipt.json")
    reference_by_id: dict[int, tuple[float, float, float]] = {}
    use_tool_by_threshold: dict[float, set[int]] = {}
    for threshold in THRESHOLDS:
        setting = f"probe_prefill_t{threshold:.1f}_{suffix}"
        relative = f"outputs/{directory}/{setting}.json"
        payload, _ = _read_json(
            root / relative, registry, f"Probe&Prefill artifact {relative}"
        )
        config = _require_mapping(payload.get("config"), f"{relative}.config")
        for key, expected in {
            "probe_protocol": probe_protocol,
            "probe_scope": probe_scope,
            "probe_training_label_seed": EXPECTED_PROBE_LABEL_SEED,
            "probe_threshold": threshold,
            "probe_temperature": formal.probe_temperature,
            "prefill_mode": "soft",
        }.items():
            _expect(config, key, expected, f"{relative}.config")
        probe_inputs = _require_mapping(
            config.get("probe_inputs_sha256"),
            f"{relative}.config.probe_inputs_sha256",
        )
        if set(probe_inputs) != expected_probe_input_names:
            raise ValueError(
                f"{relative}.config.probe_inputs_sha256 must contain exactly "
                f"{sorted(expected_probe_input_names)}"
            )
        for name in sorted(expected_probe_input_names):
            registry.register_verified(
                root / "probes" / probe_directory / name,
                _require_sha256(
                    probe_inputs[name],
                    f"{relative}.config.probe_inputs_sha256.{name}",
                ),
            )
        seed_zero_decisions: list[dict[str, Any]] = []
        use_tool_ids: set[int] = set()
        for run_index, raw_run in enumerate(payload["runs"]):
            run = _require_mapping(raw_run, f"{relative}.runs[{run_index}]")
            for row_index, raw_row in enumerate(run["rows"]):
                row = _require_mapping(
                    raw_row, f"{relative}.runs[{run_index}].rows[{row_index}]"
                )
                context = f"{relative}.runs[{run_index}].rows[{row_index}]"
                task_id = _require_int(row.get("id"), f"{context}.id")
                _expect(row, "probe_threshold", threshold, context)
                logit, probability, temperature, decision, prefill = _probe_fields(
                    row, context
                )
                if temperature != formal.probe_temperature:
                    raise ValueError(
                        f"{context}: probe temperature differs from protocol"
                    )
                expected_probability = _sigmoid_probability(logit, temperature)
                if not math.isclose(
                    probability,
                    expected_probability,
                    rel_tol=2e-15,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        f"{context}: probe_probability disagrees with "
                        "sigmoid(clip(probe_logit / probe_temperature))"
                    )
                signature = (logit, probability, temperature)
                previous = reference_by_id.setdefault(task_id, signature)
                if previous != signature:
                    raise ValueError(
                        f"Probe values vary across seed/threshold for {directory} task {task_id}"
                    )
                expected_use = probability >= threshold
                expected_decision = "use_tool" if expected_use else "no_tool"
                expected_prefill = (
                    "I need to use a tool for this question.\n"
                    if expected_use
                    else "I can solve this directly without using a tool.\n"
                )
                if decision != expected_decision or prefill != expected_prefill:
                    raise ValueError(
                        f"{context}: probe decision/prefill violates threshold"
                    )
                if run_index == 0:
                    seed_zero_decisions.append(
                        {
                            "id": task_id,
                            "probe_logit": logit,
                            "probe_probability": probability,
                            "probe_temperature": temperature,
                            "probe_threshold": threshold,
                            "probe_decision": decision,
                            "probe_prefill": prefill,
                        }
                    )
                    if expected_use:
                        use_tool_ids.add(task_id)
        expected_decisions_hash = canonical_json_sha256(seed_zero_decisions)
        _expect(
            config,
            "probe_decisions_sha256",
            expected_decisions_hash,
            f"{relative}.config",
        )
        use_tool_by_threshold[threshold] = use_tool_ids
    if len(reference_by_id) != formal.task_count:
        raise ValueError(
            f"Probe family {directory} covers {len(reference_by_id)} task IDs, "
            f"expected {formal.task_count}"
        )
    for lower, higher in zip(THRESHOLDS, THRESHOLDS[1:]):
        if not use_tool_by_threshold[higher].issubset(use_tool_by_threshold[lower]):
            raise ValueError(
                f"Probe use-tool sets are not nested for {directory}: {lower} -> {higher}"
            )
    return {
        "directory": directory,
        "probe_directory": f"probes/{probe_directory}",
        "probe_protocol": probe_protocol,
        "n_task_ids": len(reference_by_id),
        "threshold_use_tool_counts": {
            f"{threshold:.1f}": len(use_tool_by_threshold[threshold])
            for threshold in THRESHOLDS
        },
        "probe_input_hashes_verified": True,
        "probability_sigmoid_verified": True,
        "probability_logit_temperature_invariant": True,
        "decision_threshold_consistent": True,
        "use_tool_sets_nested": True,
    }


def _immutable_runs_sha(runs: Sequence[Any]) -> str:
    immutable: list[dict[str, Any]] = []
    allowed = set(RELABEL_ALLOWED_ROW_MUTATIONS)
    for raw_run in runs:
        run = dict(_require_mapping(raw_run, "relabel run"))
        rows = _require_list(run.pop("rows", None), "relabel run.rows")
        run["rows"] = [
            {
                key: copy.deepcopy(value)
                for key, value in _require_mapping(raw_row, "relabel row").items()
                if key not in allowed
            }
            for raw_row in rows
        ]
        immutable.append(run)
    return canonical_json_sha256(immutable)


def _validate_scoped_relabel(
    root: Path,
    behavior_snapshots: Mapping[str, FileSnapshot],
    registry: SnapshotRegistry,
    *,
    runtime_provenance_sha256: str,
    behavior_commit: str,
    formal: FormalProtocol,
) -> dict[str, Any]:
    receipt_path = root / "outputs/scoped_original_w2t/relabel_receipt.json"
    receipt, receipt_file = _read_json(receipt_path, registry, "scoped relabel receipt")
    for key, expected in {
        "schema_version": ACTION_SCHEMA_VERSION,
        "manifest_type": "scoped-behavior-relabel-receipt",
        "protocol_id": RELABEL_PROTOCOL,
        "model": formal.model_slug,
        "tool_scope": "scoped",
        "expected_seeds": list(formal.seeds),
        "n_task_ids": formal.task_count,
        "n_source_files": len(SCOPED_PROMPT_SETTINGS),
        "allowed_row_mutations": list(RELABEL_ALLOWED_ROW_MUTATIONS),
    }.items():
        _expect(receipt, key, expected, "relabel_receipt")
    runtime = _require_mapping(
        receipt.get("runtime_provenance"), "relabel_receipt.runtime_provenance"
    )
    _expect(
        runtime,
        "sha256",
        runtime_provenance_sha256,
        "relabel_receipt.runtime_provenance",
    )
    _expect(
        runtime,
        "project_git_commit",
        behavior_commit,
        "relabel_receipt.runtime_provenance",
    )

    label_dir = root / "labels" / formal.model_slug
    source_label_path = label_dir / "test_labels_no_reasoning_scoped.json"
    target_label_path = label_dir / "test_labels_no_reasoning_scoped_original_w2t.json"
    source_labels, source_label_file = _read_json(
        source_label_path, registry, "source scoped labels"
    )
    target_labels, target_label_file = _read_json(
        target_label_path, registry, "target original-W2T labels"
    )
    for key, path, item in (
        ("source_labels", source_label_path, source_label_file),
        ("target_labels", target_label_path, target_label_file),
    ):
        row = _require_mapping(receipt.get(key), f"relabel_receipt.{key}")
        _expect(row, "filename", path.name, f"relabel_receipt.{key}")
        _expect(row, "sha256", item.sha256, f"relabel_receipt.{key}")
    source_order, source_by_id = _validated_label_rows_by_id(
        source_labels,
        "source labels",
        formal,
        split="test",
        tool_scope="scoped",
        protocol_id=None,
    )
    target_order, target_by_id = _validated_label_rows_by_id(
        target_labels,
        "target labels",
        formal,
        split="test",
        tool_scope="scoped",
        protocol_id=RELABEL_PROTOCOL,
    )
    if source_order != target_order:
        raise ValueError("Scoped source/target label ID order differs")
    changed_task_count = 0
    for task_id in source_order:
        source_row = source_by_id[task_id]
        target_row = target_by_id[task_id]
        for key in ("category", "difficulty"):
            if source_row.get(key) != target_row.get(key):
                raise ValueError(f"Scoped label {task_id} differs on immutable {key}")
        changed_task_count += int(
            source_row.get("gold_action") != target_row.get("gold_action")
        )
    _expect(receipt, "changed_task_count", changed_task_count, "relabel_receipt")
    _expect(
        receipt,
        "task_ids_sha256",
        canonical_json_sha256(list(source_order)),
        "relabel_receipt",
    )

    raw_artifacts = _require_list(receipt.get("artifacts"), "relabel_receipt.artifacts")
    if len(raw_artifacts) != len(SCOPED_PROMPT_SETTINGS):
        raise ValueError("Relabel receipt artifact panel is incomplete")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_artifacts):
        row = _require_mapping(raw, f"relabel_receipt.artifacts[{index}]")
        setting = _require_string(
            row.get("setting"), f"relabel_receipt.artifacts[{index}].setting"
        )
        if setting in artifacts:
            raise ValueError(f"Duplicate relabel receipt setting {setting}")
        artifacts[setting] = row
    if set(artifacts) != set(SCOPED_PROMPT_SETTINGS):
        raise ValueError("Relabel receipt settings differ from scoped prompt panel")

    for setting in SCOPED_PROMPT_SETTINGS:
        source_relative = f"outputs/scoped_adapted/{setting}.json"
        target_relative = f"outputs/scoped_original_w2t/{setting}.json"
        source, _ = _read_json(
            root / source_relative, registry, f"relabel source {source_relative}"
        )
        target, _ = _read_json(
            root / target_relative, registry, f"relabel target {target_relative}"
        )
        source_snapshot = behavior_snapshots[source_relative]
        target_snapshot = behavior_snapshots[target_relative]
        artifact = artifacts[setting]
        for key, expected in {
            "source_filename": f"{setting}.json",
            "output_filename": f"{setting}.json",
            "source_sha256": source_snapshot.sha256,
            "output_sha256": target_snapshot.sha256,
        }.items():
            _expect(artifact, key, expected, f"relabel receipt artifact {setting}")
        source_config = dict(
            _require_mapping(source.get("config"), f"{source_relative}.config")
        )
        target_config = dict(
            _require_mapping(target.get("config"), f"{target_relative}.config")
        )
        for key, expected in {
            "labels_sha256": target_label_file.sha256,
            "derivation_protocol_id": RELABEL_PROTOCOL,
            "source_evaluation_sha256": source_snapshot.sha256,
            "source_labels_sha256": source_label_file.sha256,
            "target_labels_sha256": target_label_file.sha256,
        }.items():
            _expect(target_config, key, expected, f"{target_relative}.config")
        stripped_target = dict(target_config)
        for key in (
            "derivation_protocol_id",
            "source_evaluation_sha256",
            "source_labels_sha256",
            "target_labels_sha256",
        ):
            stripped_target.pop(key)
        stripped_target["labels_sha256"] = source_label_file.sha256
        if stripped_target != source_config:
            raise ValueError(
                f"{setting}: relabel changed behavior config outside allowlist"
            )

        source_runs = _require_list(source.get("runs"), f"{source_relative}.runs")
        target_runs = _require_list(target.get("runs"), f"{target_relative}.runs")
        if len(source_runs) != len(formal.seeds) or len(target_runs) != len(
            formal.seeds
        ):
            raise ValueError(
                f"{setting}: source/target relabel run counts must both equal "
                f"{len(formal.seeds)}"
            )
        source_immutable = _immutable_runs_sha(source_runs)
        target_immutable = _immutable_runs_sha(target_runs)
        if source_immutable != target_immutable:
            raise ValueError(f"{setting}: relabel changed immutable behavior fields")
        _expect(
            artifact,
            "immutable_rows_sha256",
            source_immutable,
            f"relabel artifact {setting}",
        )

        mutation_counts = {key: 0 for key in RELABEL_ALLOWED_ROW_MUTATIONS}
        source_runs_by_id: dict[str, Mapping[str, Any]] = {}
        target_runs_by_id: dict[str, Mapping[str, Any]] = {}
        for kind, raw_runs, destination in (
            ("source", source_runs, source_runs_by_id),
            ("target", target_runs, target_runs_by_id),
        ):
            for run_index, raw_run in enumerate(raw_runs):
                run = _require_mapping(raw_run, f"{kind} relabel run[{run_index}]")
                run_id = _require_string(
                    run.get("run_id"),
                    f"{kind} relabel run[{run_index}].run_id",
                )
                if run_id in destination:
                    raise ValueError(f"{setting}: duplicate {kind} run ID {run_id}")
                destination[run_id] = run
        expected_run_ids = {
            f"run_{index}_seed_{seed}" for index, seed in enumerate(formal.seeds)
        }
        if (
            set(source_runs_by_id) != expected_run_ids
            or set(target_runs_by_id) != expected_run_ids
        ):
            raise ValueError(f"{setting}: source/target relabel run IDs are incomplete")

        for run_index, seed in enumerate(formal.seeds):
            run_id = f"run_{run_index}_seed_{seed}"
            source_run_map = source_runs_by_id[run_id]
            target_run_map = target_runs_by_id[run_id]
            if set(source_run_map) != set(target_run_map):
                raise ValueError(f"{setting}: relabel changed run keys")
            if {k: v for k, v in source_run_map.items() if k != "rows"} != {
                k: v for k, v in target_run_map.items() if k != "rows"
            }:
                raise ValueError(f"{setting}: relabel changed run metadata")
            source_rows = _require_list(
                source_run_map.get("rows"), f"{setting} {run_id} source rows"
            )
            target_rows = _require_list(
                target_run_map.get("rows"), f"{setting} {run_id} target rows"
            )
            if len(source_rows) != formal.task_count or len(target_rows) != len(
                source_rows
            ):
                raise ValueError(
                    f"{setting} {run_id}: source/target row lengths must both "
                    f"equal {formal.task_count}"
                )
            source_rows_by_id: dict[int, Mapping[str, Any]] = {}
            target_rows_by_id: dict[int, Mapping[str, Any]] = {}
            source_ids: list[int] = []
            target_ids: list[int] = []
            for kind, rows, destination, ordered_ids in (
                ("source", source_rows, source_rows_by_id, source_ids),
                ("target", target_rows, target_rows_by_id, target_ids),
            ):
                for row_index, raw_row in enumerate(rows):
                    row = _require_mapping(
                        raw_row,
                        f"{setting} {run_id} {kind} rows[{row_index}]",
                    )
                    task_id = _require_int(
                        row.get("id"),
                        f"{setting} {run_id} {kind} rows[{row_index}].id",
                    )
                    if task_id in destination:
                        raise ValueError(
                            f"{setting} {run_id}: duplicate {kind} task ID {task_id}"
                        )
                    destination[task_id] = row
                    ordered_ids.append(task_id)
            if tuple(source_ids) != source_order or tuple(target_ids) != target_order:
                raise ValueError(
                    f"{setting} {run_id}: source/target row IDs/order must match labels"
                )

            for task_id in source_order:
                source_row_map = source_rows_by_id[task_id]
                target_row_map = target_rows_by_id[task_id]
                if set(source_row_map) != set(target_row_map):
                    raise ValueError(f"{setting}: relabel changed row keys")
                source_context = f"{setting} {run_id} source task {task_id}"
                target_context = f"{setting} {run_id} target task {task_id}"
                source_contract = _validate_behavior_row_against_label(
                    source_row_map, source_by_id[task_id], source_context
                )
                target_contract = _validate_behavior_row_against_label(
                    target_row_map, target_by_id[task_id], target_context
                )
                _expect(
                    source_row_map,
                    "error_type",
                    classify_action_outcome(
                        source_by_id[task_id]["gold_action"],
                        source_contract.pred_action,
                        source_contract.final_correct,
                        source_contract.invalid_tool_calls > 0,
                    ),
                    source_context,
                )
                _expect(
                    target_row_map,
                    "error_type",
                    classify_action_outcome(
                        target_by_id[task_id]["gold_action"],
                        target_contract.pred_action,
                        target_contract.final_correct,
                        target_contract.invalid_tool_calls > 0,
                    ),
                    target_context,
                )
                for key in RELABEL_ALLOWED_ROW_MUTATIONS:
                    mutation_counts[key] += int(
                        source_row_map.get(key) != target_row_map.get(key)
                    )
        for receipt_key, mutation_key in {
            "error_type_changed_rows": "error_type",
            "tool_necessary_changed_rows": "tool_necessary",
            "no_tool_correct_changed_rows": "no_tool_correct",
        }.items():
            _expect(
                artifact,
                receipt_key,
                mutation_counts[mutation_key],
                f"relabel artifact {setting}",
            )

        derivation = _require_mapping(
            target.get("derivation"), f"{target_relative}.derivation"
        )
        for key, expected in {
            "derivation_type": "scoped-behavior-gold-action-relabel",
            "protocol_id": RELABEL_PROTOCOL,
            "source_filename": f"{setting}.json",
            "source_sha256": source_snapshot.sha256,
            "source_labels_filename": source_label_path.name,
            "source_labels_sha256": source_label_file.sha256,
            "target_labels_filename": target_label_path.name,
            "target_labels_sha256": target_label_file.sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
            "project_git_commit": behavior_commit,
            "allowed_row_mutations": list(RELABEL_ALLOWED_ROW_MUTATIONS),
            "immutable_rows_sha256": source_immutable,
            "n_runs": len(formal.seeds),
            "n_task_ids": formal.task_count,
            "task_ids_sha256": canonical_json_sha256(list(source_order)),
        }.items():
            _expect(derivation, key, expected, f"{target_relative}.derivation")
    return {
        "receipt_path": receipt_file.path,
        "receipt_sha256": receipt_file.sha256,
        "n_prompt_artifacts": len(SCOPED_PROMPT_SETTINGS),
        "changed_task_count": changed_task_count,
        "source_and_target_labels_bound": True,
        "immutable_behavior_fields_preserved": True,
        "receipt_hashes_verified": True,
    }


def _default_migration_validator(
    receipt: Mapping[str, Any], context: MigrationValidationContext
) -> None:
    # Lazy import keeps lightweight unit tests and the CLI parser independent of
    # torch/sklearn; the production audit still calls the existing strict code.
    from .legacy_scoped import validate_imported_scoped_destination
    from .scripts.run_probe_prefill import validate_original_protocol_metadata

    validate_original_protocol_metadata(
        dict(receipt),
        dict(context.labels),
        model_slug=context.model_slug,
        config_sha256=context.config_sha256,
        label_seed=context.label_seed,
        task_ids=list(context.task_ids),
        n_layers=context.n_layers,
        hidden_dim=context.hidden_dim,
        probe_c=context.probe_c,
    )

    validate_imported_scoped_destination(
        receipt,
        output_root=context.root,
        model_slug=context.model_slug,
        label_seed=context.label_seed,
    )


def _validate_migration_receipt(
    root: Path,
    registry: SnapshotRegistry,
    *,
    formal: FormalProtocol,
    config_sha256: str,
    task_ids: tuple[int, ...],
    validator: MigrationValidator,
) -> dict[str, Any]:
    path = root / "probes/scoped_original_w2t/migration_receipt.json"
    receipt, item = _read_json(path, registry, "original-W2T migration receipt")
    labels_path = (
        root
        / "labels"
        / formal.model_slug
        / f"test_labels_no_reasoning_{RELABEL_PROTOCOL}.json"
    )
    labels, _ = _read_json(
        labels_path,
        registry,
        "original-W2T migration test labels",
    )
    _validated_label_rows_by_id(
        labels,
        "original-W2T migration test labels",
        formal,
        split="test",
        tool_scope="scoped",
        protocol_id=RELABEL_PROTOCOL,
    )
    validator(
        receipt,
        MigrationValidationContext(
            root=root,
            model_slug=formal.model_slug,
            config_sha256=config_sha256,
            task_ids=task_ids,
            labels=labels,
        ),
    )
    destination_hashes = _require_mapping(
        receipt.get("destination_artifact_sha256"),
        "migration_receipt.destination_artifact_sha256",
    )
    if not destination_hashes:
        raise ValueError("Migration receipt destination artifact inventory is empty")
    for relative, digest in destination_hashes.items():
        relative = _require_string(relative, "migration destination path")
        registry.register_verified(
            _resolved_run_path(root, relative, "migration destination path"),
            _require_sha256(digest, f"migration destination {relative}"),
        )
    data_hashes = _require_mapping(
        receipt.get("destination_data_files_sha256"),
        "migration_receipt.destination_data_files_sha256",
    )
    if not data_hashes:
        raise ValueError("Migration receipt destination data inventory is empty")
    for relative, digest in data_hashes.items():
        relative = _require_string(relative, "migration data path")
        registry.register_verified(
            _resolved_run_path(root, relative, "migration data path"),
            _require_sha256(digest, f"migration data {relative}"),
        )
    registry.register_verified(
        root / "data/data_manifest.json",
        _require_sha256(
            receipt.get("destination_data_manifest_sha256"),
            "migration_receipt.destination_data_manifest_sha256",
        ),
    )
    return {
        "receipt_path": item.path,
        "receipt_sha256": item.sha256,
        "destination_artifact_count": len(destination_hashes),
        "destination_data_file_count": len(data_hashes),
        "original_protocol_metadata_validation_passed": True,
        "destination_only_validation_passed": True,
    }


def _validate_output_target(root: Path, output: Path) -> Path:
    expected = root.joinpath(*AUDIT_RECEIPT_RELATIVE.parts).resolve()
    raw = Path(output)
    if raw.is_symlink():
        raise ValueError("Formal audit output must not be a symlink")
    resolved = raw.resolve()
    if resolved != expected:
        raise ValueError(f"Formal audit output must be {expected}, got {resolved}")
    if raw.exists() and not raw.is_file():
        raise ValueError("Formal audit output must be a regular file when it exists")
    return resolved


def build_formal_stage_audit(
    run_root: Path,
    *,
    config_path: Path,
    behavior_commit: str,
    statistics_commit: str,
    repository_root: Path | None = None,
    formal: FormalProtocol = FORMAL_PROTOCOL,
    migration_validator: MigrationValidator | None = None,
    scoped_menu_builder: ScopedMenuBuilder | None = None,
) -> dict[str, Any]:
    """Validate every registered formal-stage semantic binding without writing."""

    raw_root = Path(run_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError(f"run_root must be a real directory: {run_root}")
    root = raw_root.resolve()
    raw_repository = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root)
    )
    if raw_repository.is_symlink() or not raw_repository.is_dir():
        raise ValueError(f"repository_root must be a real directory: {raw_repository}")
    repository = raw_repository.resolve()
    behavior_commit = _require_commit(behavior_commit, "behavior_commit")
    statistics_commit = _require_commit(statistics_commit, "statistics_commit")
    if formal.seeds != EXPECTED_SEEDS:
        raise ValueError("Formal seed panel must remain exactly (0,1,2)")
    if formal.train_task_count <= 0 or formal.task_count <= 0:
        raise ValueError("Formal train/test task counts must be positive")
    if formal.max_rounds != EXPECTED_MAX_ROUNDS:
        raise ValueError("Formal behavior max_rounds must remain 10")
    registry = SnapshotRegistry(root, repository)
    _validate_exact_output_inventory(root)
    provenance, provenance_file, config_file = _validate_runtime_and_config(
        root,
        Path(config_path),
        repository,
        registry,
        behavior_commit=behavior_commit,
        formal=formal,
    )
    registered_config = _require_mapping(provenance["config"], "provenance.config")
    full_menu_sha256 = _require_sha256(
        provenance.get("full_menu_sha256"), "provenance.full_menu_sha256"
    )
    behavior_snapshots, task_ids = _validate_behavior_artifacts(
        root,
        registry,
        provenance_sha256=provenance_file.sha256,
        config_sha256=_require_sha256(
            registered_config.get("sha256"), "provenance.config.sha256"
        ),
        full_menu_sha256=full_menu_sha256,
        behavior_commit=behavior_commit,
        formal=formal,
        scoped_menu_builder=scoped_menu_builder or _default_scoped_menu_builder,
    )
    summaries = _validate_analysis_summaries(
        root,
        registry,
        behavior_snapshots,
        runtime_provenance_sha256=provenance_file.sha256,
        behavior_commit=behavior_commit,
        statistics_commit=statistics_commit,
        formal=formal,
    )
    p_and_p = (
        _validate_probe_prefill_family(
            root,
            registry,
            directory="fulltools",
            suffix="fulltools",
            probe_directory="fulltools",
            probe_protocol="adapted",
            probe_scope="full-adapted",
            formal=formal,
        ),
        _validate_probe_prefill_family(
            root,
            registry,
            directory="scoped_adapted",
            suffix="scoped",
            probe_directory="scoped",
            probe_protocol="adapted",
            probe_scope="scoped-adapted",
            formal=formal,
        ),
        _validate_probe_prefill_family(
            root,
            registry,
            directory="scoped_original_w2t",
            suffix="scoped_original_w2t",
            probe_directory="scoped_original_w2t",
            probe_protocol=RELABEL_PROTOCOL,
            probe_scope="scoped-original-pinned",
            formal=formal,
        ),
    )
    relabel = _validate_scoped_relabel(
        root,
        behavior_snapshots,
        registry,
        runtime_provenance_sha256=provenance_file.sha256,
        behavior_commit=behavior_commit,
        formal=formal,
    )
    migration = _validate_migration_receipt(
        root,
        registry,
        formal=formal,
        config_sha256=config_file.sha256,
        task_ids=task_ids,
        validator=migration_validator or _default_migration_validator,
    )
    registry.ensure_unchanged()
    checked_files = registry.receipt_rows()
    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "manifest_type": AUDIT_MANIFEST_TYPE,
        "audit_complete": True,
        "model": formal.model_slug,
        "code_commits": {
            "behavior": behavior_commit,
            "statistics": statistics_commit,
        },
        "registered_protocol": {
            "seeds": list(formal.seeds),
            "train_task_count": formal.train_task_count,
            "task_count": formal.task_count,
            "behavior_max_rounds": formal.max_rounds,
            "probe_temperature": formal.probe_temperature,
            "probe_thresholds": list(THRESHOLDS),
            "bootstrap_samples": formal.bootstrap_samples,
            "bootstrap_seed": formal.bootstrap_seed,
        },
        "bindings": {
            "runtime_provenance": {
                "path": provenance_file.path,
                "sha256": provenance_file.sha256,
            },
            "registered_config": {
                "path": config_file.path,
                "sha256": config_file.sha256,
            },
            "task_ids_sha256": canonical_json_sha256(list(task_ids)),
            "full_menu_sha256": full_menu_sha256,
        },
        "behavior": {
            "n_artifacts": len(behavior_snapshots),
            "n_runs": len(behavior_snapshots) * len(formal.seeds),
            "n_rows": len(behavior_snapshots) * len(formal.seeds) * formal.task_count,
            "exact_inventory_verified": True,
            "config_and_provenance_verified": True,
            "setting_modes_action_rows_and_menus_verified": True,
        },
        "statistics": {
            "protocols": {
                spec.protocol: {
                    "n_settings": len(spec.settings),
                    "n_runs": spec.expected_runs(formal),
                    "n_rows": spec.expected_rows(formal),
                    "summary_sha256": registry.snapshot(
                        root / "analysis" / spec.protocol / "summary.json"
                    ).sha256,
                }
                for spec in ANALYSIS_SPECS
            },
            "summary_count": len(summaries),
            "publication_hashes_verified": True,
        },
        "probe_prefill": list(p_and_p),
        "scoped_relabel": relabel,
        "migration": migration,
        "checked_files": checked_files,
        "totals": {
            "checked_file_count": len(checked_files),
            "checked_bytes": sum(row["bytes"] for row in checked_files),
        },
    }
    payload["checked_files_sha256"] = canonical_json_sha256(checked_files)
    return payload


def write_formal_stage_audit(
    run_root: Path,
    *,
    config_path: Path,
    output: Path,
    behavior_commit: str,
    statistics_commit: str,
    overwrite: bool = False,
    repository_root: Path | None = None,
    formal: FormalProtocol = FORMAL_PROTOCOL,
    migration_validator: MigrationValidator | None = None,
    scoped_menu_builder: ScopedMenuBuilder | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one formal-stage audit receipt."""

    root = Path(run_root).resolve()
    output_path = _validate_output_target(root, Path(output))
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite explicitly"
        )
    payload = build_formal_stage_audit(
        root,
        config_path=config_path,
        behavior_commit=behavior_commit,
        statistics_commit=statistics_commit,
        repository_root=repository_root,
        formal=formal,
        migration_validator=migration_validator,
        scoped_menu_builder=scoped_menu_builder,
    )
    atomic_write_json(output_path, payload, overwrite=overwrite)
    return payload
