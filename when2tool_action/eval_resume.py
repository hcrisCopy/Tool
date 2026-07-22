"""Strict, auditable resume handling for seeded behavior evaluations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation_contract import validate_action_row
from .io_utils import atomic_write_json, canonical_json_sha256, read_json


@dataclass(frozen=True)
class PreparedEvaluationArtifact:
    """A preflighted new/restarted target or fully validated checkpoint."""

    path: Path
    artifact: dict[str, Any]
    completed_runs: int
    needs_initialization: bool


def _require_exact_value(actual: Any, expected: Any, context: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"Resume validation failed for {context}: "
            f"expected {expected!r}, got {actual!r}"
        )


def _require_exact_keys(
    value: dict[str, Any], expected_keys: set[str], context: str
) -> None:
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Resume validation failed for {context} keys: "
            f"missing={missing}, extra={extra}"
        )


def _require_unique(values: Sequence[Any], context: str) -> None:
    try:
        unique_count = len(set(values))
    except TypeError as error:
        raise TypeError(f"{context} must contain hashable IDs") from error
    if unique_count != len(values):
        raise ValueError(f"Resume validation failed: duplicate {context}")


def validate_evaluation_artifact_for_resume(
    artifact: Any,
    *,
    template: dict[str, Any],
    task_ids: Sequence[Any],
    expected_row_fields: Mapping[Any, Mapping[str, Any]] | None = None,
) -> int:
    """Validate a checkpoint completely and return its completed run count.

    A checkpoint is resumable only when it is an exact prefix of the requested
    seed protocol.  Every completed run must contain every requested task once,
    in the original task order.  No best-effort recovery is attempted.
    """

    if not isinstance(template, dict) or template.get("runs") != []:
        raise ValueError("Evaluation resume template must be a mapping with runs=[]")
    if not isinstance(artifact, dict):
        raise TypeError("Resume target must contain a JSON object")

    expected_ids = list(task_ids)
    if not expected_ids:
        raise ValueError("Evaluation task IDs must be non-empty")
    _require_unique(expected_ids, "expected task IDs")
    if expected_row_fields is not None:
        if not isinstance(expected_row_fields, Mapping):
            raise TypeError("Expected row fields must be a task-ID mapping")
        if set(expected_row_fields) != set(expected_ids):
            raise ValueError("Expected row-field IDs must equal evaluation task IDs")
        for task_id, fields in expected_row_fields.items():
            if not isinstance(fields, Mapping) or not fields:
                raise TypeError(
                    f"Expected row fields for task {task_id!r} must be a non-empty mapping"
                )

    _require_exact_keys(artifact, set(template), "artifact")
    for key in ("schema_version", "upstream_commit"):
        _require_exact_value(artifact.get(key), template.get(key), key)

    expected_config = template.get("config")
    actual_config = artifact.get("config")
    if not isinstance(expected_config, dict):
        raise TypeError("Evaluation resume template config must be a mapping")
    if not isinstance(actual_config, dict):
        raise TypeError("Resume target config must be a mapping")
    _require_exact_keys(actual_config, set(expected_config), "config")
    for key, expected in expected_config.items():
        _require_exact_value(actual_config.get(key), expected, f"config.{key}")

    expected_task_hash = canonical_json_sha256(expected_ids)
    _require_exact_value(
        expected_config.get("task_ids_sha256"),
        expected_task_hash,
        "template config.task_ids_sha256",
    )

    seeds = expected_config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise TypeError("Evaluation resume template config.seeds must be a non-empty list")
    _require_unique(seeds, "configured seeds")
    setting = expected_config.get("setting")
    tool_scope = expected_config.get("tool_scope")

    runs = artifact.get("runs")
    if not isinstance(runs, list):
        raise TypeError("Resume target runs must be a list")
    if len(runs) > len(seeds):
        raise ValueError(
            f"Resume target has {len(runs)} runs for only {len(seeds)} configured seeds"
        )

    seen_run_ids: list[Any] = []
    seen_seeds: list[Any] = []
    expected_run_keys = {"run_id", "seed", "setting", "rows"}
    for index, run in enumerate(runs):
        context = f"runs[{index}]"
        if not isinstance(run, dict):
            raise TypeError(f"Resume target {context} must be a mapping")
        _require_exact_keys(run, expected_run_keys, context)

        expected_seed = seeds[index]
        expected_run_id = f"run_{index}_seed_{expected_seed}"
        _require_exact_value(run.get("run_id"), expected_run_id, f"{context}.run_id")
        _require_exact_value(run.get("seed"), expected_seed, f"{context}.seed")
        _require_exact_value(run.get("setting"), setting, f"{context}.setting")
        seen_run_ids.append(run.get("run_id"))
        seen_seeds.append(run.get("seed"))

        rows = run.get("rows")
        if not isinstance(rows, list):
            raise TypeError(f"Resume target {context}.rows must be a list")
        if len(rows) != len(expected_ids):
            raise ValueError(
                f"Resume validation failed for {context} task ID order/count: "
                f"expected {len(expected_ids)}, got {len(rows)}"
            )
        actual_ids: list[Any] = []
        for row_index, row in enumerate(rows):
            row_context = f"{context}.rows[{row_index}]"
            if not isinstance(row, dict):
                raise TypeError(f"Resume target {row_context} must be a mapping")
            for required_key in (
                "id",
                "schema_version",
                "run_id",
                "seed",
                "setting",
                "tool_scope",
            ):
                if required_key not in row:
                    raise ValueError(
                        f"Resume validation failed: {row_context} misses {required_key}"
                    )
            actual_ids.append(row["id"])
            _require_exact_value(
                row["schema_version"],
                template["schema_version"],
                f"{row_context}.schema_version",
            )
            _require_exact_value(
                row["run_id"], expected_run_id, f"{row_context}.run_id"
            )
            _require_exact_value(row["seed"], expected_seed, f"{row_context}.seed")
            _require_exact_value(row["setting"], setting, f"{row_context}.setting")
            _require_exact_value(
                row["tool_scope"], tool_scope, f"{row_context}.tool_scope"
            )
            validate_action_row(
                row,
                context=f"Resume target {row_context}",
                require_pred_action=True,
                strict_event_categories=True,
                require_formal_diagnostics=True,
            )
            if expected_row_fields is not None:
                expected_fields = expected_row_fields[expected_ids[row_index]]
                for field, expected_value in expected_fields.items():
                    if field not in row:
                        raise ValueError(
                            f"Resume validation failed: {row_context} misses {field}"
                        )
                    _require_exact_value(
                        row[field], expected_value, f"{row_context}.{field}"
                    )
        _require_unique(actual_ids, f"task IDs in {context}")
        if actual_ids != expected_ids:
            raise ValueError(
                f"Resume validation failed for {context} task ID order: "
                f"expected {expected_ids!r}, got {actual_ids!r}"
            )

    _require_unique(seen_run_ids, "run IDs")
    _require_unique(seen_seeds, "completed run seeds")
    return len(runs)


def prepare_evaluation_artifact(
    path: Path,
    *,
    template: dict[str, Any],
    task_ids: Sequence[Any],
    overwrite: bool,
    resume: bool,
    expected_row_fields: Mapping[Any, Mapping[str, Any]] | None = None,
) -> PreparedEvaluationArtifact:
    """Preflight a target without mutating it."""

    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    path = path.resolve()
    if path.exists():
        if overwrite:
            return PreparedEvaluationArtifact(path, deepcopy(template), 0, True)
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite {path}; pass --resume to validate and continue "
                "or --overwrite to restart"
            )
        artifact = read_json(path)
        completed = validate_evaluation_artifact_for_resume(
            artifact,
            template=template,
            task_ids=task_ids,
            expected_row_fields=expected_row_fields,
        )
        return PreparedEvaluationArtifact(path, artifact, completed, False)
    return PreparedEvaluationArtifact(path, deepcopy(template), 0, True)


def initialize_evaluation_artifacts(
    prepared_artifacts: Sequence[PreparedEvaluationArtifact],
) -> None:
    """Atomically write every preflighted empty target before model loading.

    Callers must finish preflighting their complete target set before invoking
    this function.  This two-phase API prevents a later target validation error
    from truncating an earlier target.
    """

    prepared = list(prepared_artifacts)
    paths = [item.path for item in prepared]
    _require_unique(paths, "prepared evaluation target paths")
    for item in prepared:
        if not item.needs_initialization:
            continue
        if item.completed_runs != 0 or item.artifact.get("runs") != []:
            raise ValueError(
                f"Initialization target {item.path} is not an empty evaluation artifact"
            )
    for item in prepared:
        if item.needs_initialization:
            atomic_write_json(item.path, item.artifact, overwrite=True)
