"""Audited relabeling of completed scoped-prompt behavior evaluations.

Prompt generation and tool routing do not depend on the post-hoc hard-no-tool
label protocol.  This module therefore permits reusing a completed behavior
matrix while changing only label-derived row fields, with strict source,
provenance, and immutable-payload checks.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ExperimentConfig
from .constants import (
    ACTIONS,
    EXPECTED_SPLIT_SIZES,
    SCHEMA_VERSION,
    UPSTREAM_COMMIT,
)
from .eval_resume import validate_evaluation_artifact_for_resume
from .evaluation_contract import classify_action_outcome, validate_action_row
from .io_utils import atomic_write_json, canonical_json_sha256, sha256_bytes, sha256_file
from .provenance import validate_runtime_provenance


EXPECTED_SEEDS = (0, 1, 2)
PROMPT_MODES = ("force_tool", "current", "necessary_tool", "sparse_tool", "no_tool")
REASONING_MODES = ("no_reasoning", "reasoning")
EXPECTED_SCOPED_SETTINGS = tuple(
    f"{prompt}_{reasoning}_scoped"
    for reasoning in REASONING_MODES
    for prompt in PROMPT_MODES
)
ALLOWED_ROW_MUTATIONS = (
    "gold_action",
    "error_type",
    "tool_necessary",
    "no_tool_correct",
)
RECEIPT_NAME = "relabel_receipt.json"
_PROTOCOL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,63}$")


@dataclass(frozen=True)
class LabelArtifact:
    path: Path
    sha256: str
    payload: dict[str, Any]
    task_ids: tuple[int, ...]
    rows_by_id: dict[int, dict[str, Any]]


def _read_json_object(path: Path, context: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {context}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object: {path}")
    return value, sha256_bytes(payload)


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _require_binary_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise TypeError(f"{context} must be integer 0 or 1, got {value!r}")
    return value


def _load_label_artifact(
    path: Path,
    *,
    model_slug: str,
    expected_count: int,
    protocol_id: str | None,
) -> LabelArtifact:
    payload, digest = _read_json_object(path, "label artifact")
    required_top = {
        "schema_version",
        "upstream_commit",
        "model",
        "split",
        "seed",
        "prompt_mode",
        "reasoning_mode",
        "tool_scope",
        "n",
        "rows",
    }
    missing = sorted(required_top - set(payload))
    if missing:
        raise ValueError(f"{path}: label artifact is missing {missing}")
    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": model_slug,
        "split": "test",
        "seed": EXPECTED_SEEDS[0],
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "scoped",
    }.items():
        if payload[key] != expected:
            raise ValueError(f"{path}: {key}={payload[key]!r}, expected {expected!r}")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise TypeError(f"{path}: seed must be an integer")
    if protocol_id is not None and payload.get("protocol_id") != protocol_id:
        raise ValueError(
            f"{path}: protocol_id={payload.get('protocol_id')!r}, "
            f"expected {protocol_id!r}"
        )
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} label rows, got "
            f"{len(rows) if isinstance(rows, list) else type(rows).__name__}"
        )
    if payload["n"] != expected_count:
        raise ValueError(f"{path}: top-level n must equal {expected_count}")

    task_ids: list[int] = []
    rows_by_id: dict[int, dict[str, Any]] = {}
    required_row = {
        "id",
        "split",
        "difficulty",
        "category",
        "tool_necessary",
        "gold_action",
        "seed",
        "prompt_mode",
        "reasoning_mode",
        "tool_scope",
    }
    for index, raw_row in enumerate(rows):
        context = f"{path} rows[{index}]"
        if not isinstance(raw_row, dict):
            raise TypeError(f"{context} must be a JSON object")
        missing_row = sorted(required_row - set(raw_row))
        if missing_row:
            raise ValueError(f"{context} is missing {missing_row}")
        task_id = raw_row["id"]
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise TypeError(f"{context}.id must be an integer")
        if task_id in rows_by_id:
            raise ValueError(f"{path}: duplicate label task ID {task_id}")
        category = raw_row["category"]
        if category not in {"A", "B", "C"}:
            raise ValueError(f"{context}.category is invalid: {category!r}")
        necessary = _require_binary_int(
            raw_row["tool_necessary"], f"{context}.tool_necessary"
        )
        expected_action = category if necessary else "NONE"
        if raw_row["gold_action"] != expected_action:
            raise ValueError(
                f"{context}.gold_action={raw_row['gold_action']!r}, "
                f"expected {expected_action!r}"
            )
        if "no_tool_correct" in raw_row:
            no_tool_correct = _require_binary_int(
                raw_row["no_tool_correct"], f"{context}.no_tool_correct"
            )
            if no_tool_correct != 1 - necessary:
                raise ValueError(f"{context}.no_tool_correct disagrees with tool_necessary")
        for key, expected in {
            "split": "test",
            "seed": payload["seed"],
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": "scoped",
        }.items():
            if raw_row[key] != expected:
                raise ValueError(
                    f"{context}.{key}={raw_row[key]!r}, expected {expected!r}"
                )
        _require_nonempty_string(raw_row["difficulty"], f"{context}.difficulty")
        if protocol_id is not None and raw_row.get("label_protocol") != protocol_id:
            raise ValueError(
                f"{context}.label_protocol={raw_row.get('label_protocol')!r}, "
                f"expected {protocol_id!r}"
            )
        task_ids.append(task_id)
        rows_by_id[task_id] = raw_row
    return LabelArtifact(
        path=path.resolve(),
        sha256=digest,
        payload=payload,
        task_ids=tuple(task_ids),
        rows_by_id=rows_by_id,
    )


def _immutable_runs_payload(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    immutable_runs: list[dict[str, Any]] = []
    allowed = set(ALLOWED_ROW_MUTATIONS)
    for run in runs:
        immutable_run = {key: deepcopy(value) for key, value in run.items() if key != "rows"}
        immutable_run["rows"] = [
            {key: deepcopy(value) for key, value in row.items() if key not in allowed}
            for row in run["rows"]
        ]
        immutable_runs.append(immutable_run)
    return immutable_runs


def _validate_label_pair(source: LabelArtifact, target: LabelArtifact) -> int:
    if source.task_ids != target.task_ids:
        raise ValueError("Source and target label task IDs/order differ")
    changed = 0
    for task_id in source.task_ids:
        source_row = source.rows_by_id[task_id]
        target_row = target.rows_by_id[task_id]
        for key in ("category", "difficulty"):
            if source_row[key] != target_row[key]:
                raise ValueError(f"Label task {task_id} disagrees on {key}")
        changed += int(source_row["gold_action"] != target_row["gold_action"])
    if changed == 0:
        raise ValueError("Source and target labels do not change any gold action")
    return changed


def _relabel_one(
    source_path: Path,
    *,
    source_labels: LabelArtifact,
    target_labels: LabelArtifact,
    protocol_id: str,
    model_slug: str,
    runtime_provenance: Mapping[str, Any],
    expected_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source, source_sha256 = _read_json_object(source_path, "evaluation artifact")
    if set(source) != {"schema_version", "upstream_commit", "config", "runs"}:
        raise ValueError(
            f"{source_path}: source must be a direct evaluation artifact with exact "
            "top-level keys schema_version/upstream_commit/config/runs"
        )
    if source["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{source_path}: schema_version mismatch")
    if source["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError(f"{source_path}: upstream_commit mismatch")
    config = source["config"]
    if not isinstance(config, dict):
        raise TypeError(f"{source_path}: config must be a JSON object")
    required_config = {
        "model",
        "setting",
        "tool_scope",
        "seeds",
        "task_ids_sha256",
        "labels_sha256",
        "runtime_provenance_sha256",
        "project_git_commit",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise ValueError(f"{source_path}: config is missing {missing_config}")
    for key, expected in {
        "model": model_slug,
        "tool_scope": "scoped",
        "seeds": list(expected_seeds),
        "labels_sha256": source_labels.sha256,
        "runtime_provenance_sha256": runtime_provenance["sha256"],
        "project_git_commit": runtime_provenance["git_commit"],
    }.items():
        if config[key] != expected:
            raise ValueError(
                f"{source_path}: config.{key}={config[key]!r}, expected {expected!r}"
            )
    if config.get("smoke") is not False:
        raise ValueError(f"{source_path}: formal relabeling requires smoke=false")
    setting = _require_nonempty_string(config["setting"], f"{source_path}: setting")
    if source_path.name != f"{setting}.json":
        raise ValueError(
            f"{source_path}: filename must be the registered setting name {setting}.json"
        )

    task_ids = list(source_labels.task_ids)
    template = deepcopy(source)
    template["runs"] = []
    expected_source_fields = {
        task_id: {"gold_action": source_labels.rows_by_id[task_id]["gold_action"]}
        for task_id in task_ids
    }
    completed = validate_evaluation_artifact_for_resume(
        source,
        template=template,
        task_ids=task_ids,
        expected_row_fields=expected_source_fields,
    )
    if completed != len(expected_seeds):
        raise ValueError(
            f"{source_path}: expected {len(expected_seeds)} complete runs, got {completed}"
        )

    output = deepcopy(source)
    derivation_keys = {
        "derivation_protocol_id",
        "source_evaluation_sha256",
        "source_labels_sha256",
        "target_labels_sha256",
    }
    collisions = sorted(derivation_keys & set(output["config"]))
    if collisions:
        raise ValueError(f"{source_path}: source config already has derivation keys {collisions}")

    error_type_changes = 0
    optional_field_changes = {"tool_necessary": 0, "no_tool_correct": 0}
    for run_index, run in enumerate(output["runs"]):
        for row_index, row in enumerate(run["rows"]):
            task_id = row["id"]
            source_label = source_labels.rows_by_id[task_id]
            target_label = target_labels.rows_by_id[task_id]
            context = f"{source_path} runs[{run_index}].rows[{row_index}]"
            if row.get("category") != source_label["category"]:
                raise ValueError(f"{context}.category disagrees with source labels")
            if "difficulty" in row and row["difficulty"] != source_label["difficulty"]:
                raise ValueError(f"{context}.difficulty disagrees with source labels")
            if "error_type" not in row:
                raise ValueError(f"{context} is missing required derived error_type")
            if "tool_necessary" in row:
                source_necessary = _require_binary_int(
                    row["tool_necessary"], f"{context}.tool_necessary"
                )
                if source_necessary != source_label["tool_necessary"]:
                    raise ValueError(f"{context}.tool_necessary disagrees with source labels")
            if "no_tool_correct" in row:
                source_no_tool = _require_binary_int(
                    row["no_tool_correct"], f"{context}.no_tool_correct"
                )
                if source_no_tool != 1 - source_label["tool_necessary"]:
                    raise ValueError(f"{context}.no_tool_correct disagrees with source labels")

            contract = validate_action_row(
                row,
                context=context,
                require_pred_action=True,
                strict_event_categories=True,
                require_formal_diagnostics=True,
            )
            new_gold = target_label["gold_action"]
            new_error = classify_action_outcome(
                new_gold,
                contract.pred_action,
                contract.final_correct,
                contract.invalid_tool_calls > 0,
            )
            error_type_changes += int(row["error_type"] != new_error)
            row["gold_action"] = new_gold
            row["error_type"] = new_error
            if "tool_necessary" in row:
                optional_field_changes["tool_necessary"] += int(
                    row["tool_necessary"] != target_label["tool_necessary"]
                )
                row["tool_necessary"] = target_label["tool_necessary"]
            if "no_tool_correct" in row:
                target_no_tool_correct = 1 - target_label["tool_necessary"]
                optional_field_changes["no_tool_correct"] += int(
                    row["no_tool_correct"] != target_no_tool_correct
                )
                row["no_tool_correct"] = target_no_tool_correct
            validate_action_row(
                row,
                context=f"derived {context}",
                require_pred_action=True,
                strict_event_categories=True,
                require_formal_diagnostics=True,
            )

    source_immutable = canonical_json_sha256(_immutable_runs_payload(source["runs"]))
    output_immutable = canonical_json_sha256(_immutable_runs_payload(output["runs"]))
    if output_immutable != source_immutable:
        raise AssertionError("Relabeling changed a row field outside the allowlist")

    output["config"].update(
        {
            "labels_sha256": target_labels.sha256,
            "derivation_protocol_id": protocol_id,
            "source_evaluation_sha256": source_sha256,
            "source_labels_sha256": source_labels.sha256,
            "target_labels_sha256": target_labels.sha256,
        }
    )
    output["derivation"] = {
        "derivation_type": "scoped-behavior-gold-action-relabel",
        "protocol_id": protocol_id,
        "source_filename": source_path.name,
        "source_sha256": source_sha256,
        "source_labels_filename": source_labels.path.name,
        "source_labels_sha256": source_labels.sha256,
        "target_labels_filename": target_labels.path.name,
        "target_labels_sha256": target_labels.sha256,
        "runtime_provenance_sha256": runtime_provenance["sha256"],
        "project_git_commit": runtime_provenance["git_commit"],
        "allowed_row_mutations": list(ALLOWED_ROW_MUTATIONS),
        "immutable_rows_sha256": source_immutable,
        "n_runs": len(expected_seeds),
        "n_task_ids": len(task_ids),
        "task_ids_sha256": canonical_json_sha256(task_ids),
    }
    output_template = deepcopy(output)
    output_template["runs"] = []
    expected_target_fields = {
        task_id: {"gold_action": target_labels.rows_by_id[task_id]["gold_action"]}
        for task_id in task_ids
    }
    if (
        validate_evaluation_artifact_for_resume(
            output,
            template=output_template,
            task_ids=task_ids,
            expected_row_fields=expected_target_fields,
        )
        != len(expected_seeds)
    ):
        raise AssertionError("Derived evaluation lost a complete run")
    summary = {
        "setting": setting,
        "source_filename": source_path.name,
        "source_sha256": source_sha256,
        "output_filename": source_path.name,
        "immutable_rows_sha256": source_immutable,
        "error_type_changed_rows": error_type_changes,
        "tool_necessary_changed_rows": optional_field_changes["tool_necessary"],
        "no_tool_correct_changed_rows": optional_field_changes["no_tool_correct"],
    }
    return output, summary


def _publish_staged_files(
    stage_dir: Path,
    output_dir: Path,
    names: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in names]
    conflicts = [path for path in targets if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite relabel outputs: "
            + ", ".join(str(path) for path in conflicts)
        )
    if any(path.exists() and not path.is_file() for path in conflicts):
        raise ValueError("Relabel output targets must be regular files")

    backup_dir = Path(tempfile.mkdtemp(prefix=".relabel-backup-", dir=output_dir.parent))
    backed_up: list[tuple[Path, Path]] = []
    published: list[Path] = []
    cleanup_backup = True
    try:
        for target in conflicts:
            backup = backup_dir / target.name
            os.replace(target, backup)
            backed_up.append((target, backup))
        for name, target in zip(names, targets):
            os.replace(stage_dir / name, target)
            published.append(target)
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        for target in reversed(published):
            try:
                if target.is_file():
                    target.unlink()
            except OSError as error:
                rollback_errors.append(f"remove {target}: {error}")
        for target, backup in backed_up:
            try:
                if backup.is_file():
                    os.replace(backup, target)
            except OSError as error:
                rollback_errors.append(f"restore {target}: {error}")
        if rollback_errors:
            cleanup_backup = False
            raise RuntimeError(
                "Relabel publication failed and rollback was incomplete; backups "
                f"were retained at {backup_dir}: {rollback_errors}"
            ) from publish_error
        raise
    finally:
        if cleanup_backup:
            shutil.rmtree(backup_dir, ignore_errors=False)


def relabel_scoped_outputs(
    config: ExperimentConfig,
    *,
    input_paths: Sequence[Path],
    source_labels_path: Path,
    target_labels_path: Path,
    output_dir: Path,
    protocol_id: str,
    overwrite: bool = False,
    expected_settings: Sequence[str] = EXPECTED_SCOPED_SETTINGS,
    expected_task_count: int = EXPECTED_SPLIT_SIZES["test"],
) -> dict[str, Any]:
    """Validate, stage, and transactionally publish a relabeled prompt matrix."""

    if not _PROTOCOL_PATTERN.fullmatch(protocol_id):
        raise ValueError(f"Invalid protocol ID: {protocol_id!r}")
    if expected_task_count <= 0:
        raise ValueError("expected_task_count must be positive")
    normalized_inputs = [Path(path).resolve() for path in input_paths]
    if not normalized_inputs:
        raise ValueError("At least one evaluation input is required")
    if len(normalized_inputs) != len(set(normalized_inputs)):
        raise ValueError("Duplicate evaluation input paths")
    basenames = [path.name for path in normalized_inputs]
    if len(basenames) != len(set(basenames)):
        raise ValueError("Evaluation inputs must have unique basenames")
    expected_setting_set = set(expected_settings)
    if not expected_setting_set or len(expected_setting_set) != len(expected_settings):
        raise ValueError("Expected scoped settings must be non-empty and unique")

    runtime_provenance = validate_runtime_provenance(config)
    source_labels = _load_label_artifact(
        Path(source_labels_path),
        model_slug=config.model.slug,
        expected_count=expected_task_count,
        protocol_id=None,
    )
    target_labels = _load_label_artifact(
        Path(target_labels_path),
        model_slug=config.model.slug,
        expected_count=expected_task_count,
        protocol_id=protocol_id,
    )
    changed_task_count = _validate_label_pair(source_labels, target_labels)

    outputs: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    settings: set[str] = set()
    for source_path in normalized_inputs:
        output, summary = _relabel_one(
            source_path,
            source_labels=source_labels,
            target_labels=target_labels,
            protocol_id=protocol_id,
            model_slug=config.model.slug,
            runtime_provenance=runtime_provenance,
            expected_seeds=EXPECTED_SEEDS,
        )
        setting = summary["setting"]
        if setting in settings:
            raise ValueError(f"Duplicate evaluation setting {setting!r}")
        settings.add(setting)
        outputs[source_path.name] = output
        summaries.append(summary)
    if settings != expected_setting_set:
        raise ValueError(
            f"Scoped prompt setting panel mismatch: "
            f"missing={sorted(expected_setting_set - settings)}, "
            f"extra={sorted(settings - expected_setting_set)}"
        )

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output directory path is not a directory: {output_dir}")
    target_paths = [output_dir / name for name in [*outputs, RECEIPT_NAME]]
    source_set = set(normalized_inputs) | {source_labels.path, target_labels.path}
    aliases = [path for path in target_paths if path.resolve() in source_set]
    if aliases:
        raise ValueError(f"Output targets alias source inputs: {aliases}")
    conflicts = [path for path in target_paths if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite relabel outputs: "
            + ", ".join(str(path) for path in conflicts)
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".relabel-stage-", dir=output_dir.parent))
    names = [*outputs, RECEIPT_NAME]
    try:
        for name, artifact in outputs.items():
            atomic_write_json(stage_dir / name, artifact, overwrite=False)
        for summary in summaries:
            summary["output_sha256"] = sha256_file(
                stage_dir / summary["output_filename"]
            )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "scoped-behavior-relabel-receipt",
            "protocol_id": protocol_id,
            "model": config.model.slug,
            "tool_scope": "scoped",
            "expected_seeds": list(EXPECTED_SEEDS),
            "n_task_ids": expected_task_count,
            "task_ids_sha256": canonical_json_sha256(source_labels.task_ids),
            "n_source_files": len(outputs),
            "changed_task_count": changed_task_count,
            "allowed_row_mutations": list(ALLOWED_ROW_MUTATIONS),
            "source_labels": {
                "filename": source_labels.path.name,
                "sha256": source_labels.sha256,
            },
            "target_labels": {
                "filename": target_labels.path.name,
                "sha256": target_labels.sha256,
            },
            "runtime_provenance": {
                "filename": Path(runtime_provenance["path"]).name,
                "sha256": runtime_provenance["sha256"],
                "project_git_commit": runtime_provenance["git_commit"],
            },
            "artifacts": sorted(summaries, key=lambda item: item["setting"]),
        }
        atomic_write_json(stage_dir / RECEIPT_NAME, receipt, overwrite=False)

        for path, expected_hash in [
            *[(path, summary["source_sha256"]) for path, summary in zip(normalized_inputs, summaries)],
            (source_labels.path, source_labels.sha256),
            (target_labels.path, target_labels.sha256),
            (Path(runtime_provenance["path"]), runtime_provenance["sha256"]),
        ]:
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Input changed during relabel staging: {path}")
        _publish_staged_files(stage_dir, output_dir, names, overwrite=overwrite)
        return receipt
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=False)
