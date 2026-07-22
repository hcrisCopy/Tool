"""Strict action-level statistics for full-tool When2Tool evaluations.

The public entry point is :func:`collect_action_statistics`.  Evaluation JSON
must use one of these two explicit shapes::

    {"runs": [{"run_id": "0", "seed": 0, "setting": "current",
                "rows": [...]}]}

or a single run object::

    {"run_id": "0", "seed": 0, "setting": "current", "rows": [...]}

Every row repeats ``run_id``, ``seed`` and ``setting``.  Repetition is
intentional: it makes each persisted row independently auditable, and this
module rejects disagreements between run-level and row-level metadata.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score


ACTIONS = ("NONE", "A", "B", "C")
PREDICTIONS = (*ACTIONS, "INVALID")
TOOL_ACTIONS = ("A", "B", "C")
OUTCOME_ORDER = (
    "success",
    "direct_answer_wrong",
    "answer_wrong",
    "over_call",
    "under_call",
    "wrong_category",
    "invalid_tool",
)
ERROR_HIERARCHY = (
    "invalid_tool (any routed event has an unknown category)",
    "over_call (gold NONE, at least one valid call)",
    "under_call (gold A/B/C, zero calls)",
    "wrong_category (gold A/B/C, first valid call has another category)",
    "direct_answer_wrong/answer_wrong (action is correct but final answer is wrong)",
    "success",
)
OPTIONAL_BOOLEAN_FIELDS = (
    "first_env_correct",
    "exact_tool_allowed",
    "invalid_args",
)
BOOTSTRAP_METRICS = (
    "final_accuracy",
    "action_accuracy",
    "avg_tool_calls",
    "total_tool_calls_per_run",
)
SCHEMA_VERSION = "when2tool_action_stats.v1"

METRIC_DEFINITIONS = {
    "final_accuracy": "Fraction of tasks whose final answer is correct.",
    "total_tool_calls": "Sum of executed/routed tool calls in the run.",
    "avg_tool_calls": "Mean executed/routed tool calls per task.",
    "action_accuracy": "Accuracy of the first executed tool category, with zero calls mapped to NONE.",
    "balanced_accuracy": "Unweighted mean recall over gold action classes present in the run.",
    "macro_f1": "Macro F1 over gold classes NONE/A/B/C; INVALID predictions count as misses.",
    "toolneed_f1": "Binary F1 for tool-needed (A/B/C) versus NONE; INVALID is a tool attempt.",
    "overcall_rate": "Among gold NONE tasks, fraction with one or more calls.",
    "under_call_rate": "Among gold A/B/C tasks, fraction with zero calls.",
    "wrong_category_rate": "Among gold A/B/C tasks, fraction whose first call is another valid category.",
    "invalid_tool_rate": "Fraction of tasks with an INVALID category anywhere in the routed call sequence.",
    "majority_accuracy_baseline": "Accuracy of always predicting the most frequent gold action in that run.",
    "prior_matched_expected_accuracy": "Expected accuracy of sampling predictions from the empirical gold prior, sum_c p(c)^2.",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation input does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object, got {type(value).__name__}")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string, got {value!r}")
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}, got {value}")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a JSON boolean, got {value!r}")
    return value


def _canonical_identifier(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{context} must be a string or integer, got {value!r}")
    result = str(value)
    if not result:
        raise ValueError(f"{context} must not be empty")
    return result


def _run_objects(payload: Any, path: Path) -> list[Mapping[str, Any]]:
    root = _require_mapping(payload, f"Top level of {path}")
    if "runs" in root:
        runs = root["runs"]
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"{path}: 'runs' must be a non-empty list")
        return [
            _require_mapping(run, f"{path} runs[{index}]")
            for index, run in enumerate(runs)
        ]
    required = {"run_id", "seed", "setting", "rows"}
    missing = sorted(required - set(root))
    if missing:
        raise ValueError(
            f"{path}: expected a top-level 'runs' list or one run object; missing {missing}"
        )
    return [root]


def _optional_event_boolean(
    row: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    field: str,
    context: str,
) -> bool | None:
    row_value: bool | None = None
    if field in row:
        row_value = _require_bool(row[field], f"{context}.{field}")

    event_values: list[bool] = []
    if field == "invalid_args":
        for index, event in enumerate(events):
            if field in event:
                event_values.append(
                    _require_bool(
                        event[field], f"{context}.routed_tool_events[{index}].{field}"
                    )
                )
            elif "arguments_valid" in event:
                arguments_valid = _require_bool(
                    event["arguments_valid"],
                    f"{context}.routed_tool_events[{index}].arguments_valid",
                )
                event_values.append(not arguments_valid)
        event_value = any(event_values) if event_values else None
    else:
        event_value = None
        if events and field in events[0]:
            event_value = _require_bool(
                events[0][field], f"{context}.routed_tool_events[0].{field}"
            )

    if row_value is not None and event_value is not None and row_value != event_value:
        raise ValueError(
            f"{context}.{field}={row_value} disagrees with routed event value {event_value}"
        )
    return row_value if row_value is not None else event_value


def _classify_outcome(
    gold_action: str,
    pred_action: str,
    final_correct: bool,
    has_invalid_tool: bool,
) -> str:
    """Apply the preregistered, mutually exclusive error hierarchy."""

    if has_invalid_tool:
        return "invalid_tool"
    if gold_action == "NONE":
        if pred_action == "NONE":
            return "success" if final_correct else "direct_answer_wrong"
        return "over_call"
    if pred_action == "NONE":
        return "under_call"
    if pred_action != gold_action:
        return "wrong_category"
    return "success" if final_correct else "answer_wrong"


def _derive_row(
    row: Mapping[str, Any],
    run: Mapping[str, Any],
    source_path: Path,
    row_index: int,
) -> dict[str, Any]:
    context = f"{source_path} run={run.get('run_id')!r} rows[{row_index}]"
    required = {
        "id",
        "run_id",
        "seed",
        "gold_action",
        "routed_tool_events",
        "tool_calls",
        "final_correct",
        "setting",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")

    run_id = _canonical_identifier(run.get("run_id"), f"{context} run.run_id")
    seed = _require_int(run.get("seed"), f"{context} run.seed")
    setting = _require_nonempty_string(run.get("setting"), f"{context} run.setting")
    row_run_id = _canonical_identifier(row["run_id"], f"{context}.run_id")
    row_seed = _require_int(row["seed"], f"{context}.seed")
    row_setting = _require_nonempty_string(row["setting"], f"{context}.setting")
    if (row_run_id, row_seed, row_setting) != (run_id, seed, setting):
        raise ValueError(
            f"{context}: row metadata {(row_run_id, row_seed, row_setting)!r} does not "
            f"match run metadata {(run_id, seed, setting)!r}"
        )

    task_id = _canonical_identifier(row["id"], f"{context}.id")
    gold_action = row["gold_action"]
    if gold_action not in ACTIONS:
        raise ValueError(
            f"{context}.gold_action must be one of {ACTIONS}, got {gold_action!r}"
        )
    final_correct = _require_bool(row["final_correct"], f"{context}.final_correct")
    tool_calls = _require_int(row["tool_calls"], f"{context}.tool_calls", minimum=0)

    raw_events = row["routed_tool_events"]
    if not isinstance(raw_events, list):
        raise TypeError(f"{context}.routed_tool_events must be a list")
    events = [
        _require_mapping(event, f"{context}.routed_tool_events[{index}]")
        for index, event in enumerate(raw_events)
    ]
    if len(events) != tool_calls:
        raise ValueError(
            f"{context}: len(routed_tool_events)={len(events)} != tool_calls={tool_calls}"
        )

    category_sequence: list[str] = []
    for index, event in enumerate(events):
        if "category" not in event:
            raise ValueError(
                f"{context}.routed_tool_events[{index}] is missing 'category'"
            )
        category = event["category"]
        category_sequence.append(category if category in TOOL_ACTIONS else "INVALID")
    pred_action = "NONE" if not category_sequence else category_sequence[0]
    has_invalid_tool = "INVALID" in category_sequence
    mixed_calls = len(set(category_sequence)) > 1
    outcome = _classify_outcome(
        gold_action, pred_action, final_correct, has_invalid_tool
    )

    result: dict[str, Any] = {
        "source_file": str(source_path),
        "id": task_id,
        "run_id": run_id,
        "seed": seed,
        "setting": setting,
        "gold_action": gold_action,
        "pred_action": pred_action,
        "final_correct": final_correct,
        "tool_calls": tool_calls,
        "category_sequence": json.dumps(category_sequence, ensure_ascii=False),
        "category_sequence_text": ">".join(category_sequence)
        if category_sequence
        else "NONE",
        "mixed_calls": mixed_calls,
        "has_invalid_tool": has_invalid_tool,
        "outcome": outcome,
        "routed_tool_events": json.dumps(
            raw_events, ensure_ascii=False, sort_keys=True
        ),
    }
    for field in OPTIONAL_BOOLEAN_FIELDS:
        result[field] = _optional_event_boolean(row, events, field, context)
    return result


def load_evaluation_outputs(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Load, validate and derive action records from one or more JSON files."""

    if not paths:
        raise ValueError("At least one evaluation output path is required")
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = _read_json(path)
        for run_index, run in enumerate(_run_objects(payload, path)):
            for field in ("run_id", "seed", "setting", "rows"):
                if field not in run:
                    raise ValueError(f"{path} runs[{run_index}] is missing {field!r}")
            rows = run["rows"]
            if not isinstance(rows, list) or not rows:
                raise ValueError(
                    f"{path} runs[{run_index}].rows must be a non-empty list"
                )
            for row_index, raw_row in enumerate(rows):
                row = _require_mapping(
                    raw_row, f"{path} runs[{run_index}].rows[{row_index}]"
                )
                records.append(_derive_row(row, run, path, row_index))

    frame = pd.DataFrame.from_records(records)
    _validate_evaluation_frame(frame)
    for field in OPTIONAL_BOOLEAN_FIELDS:
        frame[field] = frame[field].astype("boolean")
    return frame.sort_values(["setting", "run_id", "id"], kind="stable").reset_index(
        drop=True
    )


def _validate_evaluation_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Evaluation outputs contain no rows")
    duplicate_keys = frame.duplicated(["setting", "run_id", "id"], keep=False)
    if duplicate_keys.any():
        examples = frame.loc[duplicate_keys, ["setting", "run_id", "id"]].head(10)
        raise ValueError(
            "Duplicate (setting, run_id, id) rows:\n" + examples.to_string(index=False)
        )
    for (setting, run_id), group in frame.groupby(["setting", "run_id"], sort=True):
        seeds = group["seed"].unique().tolist()
        if len(seeds) != 1:
            raise ValueError(f"Run {(setting, run_id)!r} has multiple seeds: {seeds}")

    run_groups = list(frame.groupby(["setting", "run_id"], sort=True))
    reference_key, reference = run_groups[0]
    reference_ids = set(reference["id"])
    for key, group in run_groups[1:]:
        ids = set(group["id"])
        if ids != reference_ids:
            missing = sorted(reference_ids - ids)[:10]
            extra = sorted(ids - reference_ids)[:10]
            raise ValueError(
                f"Run {key!r} does not have the same ID set as {reference_key!r}; "
                f"missing(first 10)={missing}, extra(first 10)={extra}"
            )

    paired_duplicates = frame.duplicated(["setting", "seed", "id"], keep=False)
    if paired_duplicates.any():
        examples = frame.loc[
            paired_duplicates, ["setting", "seed", "id", "run_id"]
        ].head(10)
        raise ValueError(
            "Paired statistics require unique (setting, seed, id) rows:\n"
            + examples.to_string(index=False)
        )


def _safe_rate(numerator: int | float, denominator: int, context: str) -> float:
    if denominator == 0:
        return math.nan
    value = float(numerator) / denominator
    if not 0.0 <= value <= 1.0:
        raise AssertionError(f"Invalid rate for {context}: {value}")
    return value


def _per_run_metric_row(group: pd.DataFrame) -> dict[str, Any]:
    setting_values = group["setting"].unique()
    run_values = group["run_id"].unique()
    seed_values = group["seed"].unique()
    if len(setting_values) != 1 or len(run_values) != 1 or len(seed_values) != 1:
        raise AssertionError("Per-run metric group contains mixed run metadata")

    gold = group["gold_action"].to_numpy(dtype=object)
    pred = group["pred_action"].to_numpy(dtype=object)
    n_tasks = len(group)
    recalls: dict[str, float] = {}
    for action in ACTIONS:
        mask = gold == action
        recalls[action] = _safe_rate(
            np.sum(pred[mask] == action), int(mask.sum()), f"recall_{action}"
        )
    present_recalls = [value for value in recalls.values() if not math.isnan(value)]
    balanced_accuracy = float(np.mean(present_recalls))

    gold_need = gold != "NONE"
    pred_need = pred != "NONE"
    gold_none = ~gold_need
    prior = pd.Series(gold).value_counts(normalize=True)
    row: dict[str, Any] = {
        "setting": setting_values[0],
        "run_id": run_values[0],
        "seed": int(seed_values[0]),
        "n_tasks": n_tasks,
        "final_accuracy": float(group["final_correct"].mean()),
        "total_tool_calls": int(group["tool_calls"].sum()),
        "avg_tool_calls": float(group["tool_calls"].mean()),
        "action_accuracy": float(np.mean(gold == pred)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(
            f1_score(gold, pred, labels=list(ACTIONS), average="macro", zero_division=0)
        ),
        "toolneed_f1": float(f1_score(gold_need, pred_need, zero_division=0)),
        "overcall_rate": _safe_rate(
            np.sum(pred[gold_none] != "NONE"), int(gold_none.sum()), "overcall"
        ),
        "under_call_rate": _safe_rate(
            np.sum(pred[gold_need] == "NONE"), int(gold_need.sum()), "under-call"
        ),
        "wrong_category_rate": _safe_rate(
            np.sum(
                np.isin(pred[gold_need], TOOL_ACTIONS)
                & (pred[gold_need] != gold[gold_need])
            ),
            int(gold_need.sum()),
            "wrong-category",
        ),
        "invalid_tool_rate": float(group["has_invalid_tool"].mean()),
        "mixed_call_rate": float(group["mixed_calls"].mean()),
        "majority_accuracy_baseline": float(prior.max()),
        "prior_matched_expected_accuracy": float(np.square(prior.to_numpy()).sum()),
    }
    for action in ACTIONS:
        row[f"recall_{action}"] = recalls[action]
    for action in TOOL_ACTIONS:
        mask = gold == action
        denominator = int(mask.sum())
        row[f"under_call_{action}"] = _safe_rate(
            np.sum(pred[mask] == "NONE"), denominator, f"under_call_{action}"
        )
        row[f"wrong_category_{action}"] = _safe_rate(
            np.sum(np.isin(pred[mask], TOOL_ACTIONS) & (pred[mask] != action)),
            denominator,
            f"wrong_category_{action}",
        )
    none_mask = gold == "NONE"
    row["overcall_NONE"] = _safe_rate(
        np.sum(pred[none_mask] != "NONE"), int(none_mask.sum()), "overcall_NONE"
    )

    for field in OPTIONAL_BOOLEAN_FIELDS:
        known = group[field].notna()
        if field in {"first_env_correct", "exact_tool_allowed"}:
            # These are properties of the first executed call.  Evaluators may
            # serialize False for no-call rows; such rows are not observations
            # of either quantity and must not enter the denominator.
            known &= group["tool_calls"] > 0
        if known.any():
            row[f"{field}_n"] = int(known.sum())
            row[f"{field}_rate"] = float(group.loc[known, field].astype(bool).mean())
    return row


def compute_per_run_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute all requested metrics separately for every experimental run."""

    _validate_evaluation_frame(frame)
    rows = [
        _per_run_metric_row(group)
        for _, group in frame.groupby(["setting", "run_id"], sort=True)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["setting", "run_id"], kind="stable")
        .reset_index(drop=True)
    )


def summarize_run_metrics(per_run: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy mean +/- population-SD table grouped by setting."""

    keys = {"setting", "run_id", "seed"}
    metric_columns = [
        column
        for column in per_run.columns
        if column not in keys and pd.api.types.is_numeric_dtype(per_run[column])
    ]
    rows: list[dict[str, Any]] = []
    for setting, group in per_run.groupby("setting", sort=True):
        for metric in metric_columns:
            values = group[metric].dropna().astype(float).to_numpy()
            if not len(values):
                continue
            mean = float(values.mean())
            population_sd = float(values.std(ddof=0))
            rows.append(
                {
                    "setting": setting,
                    "n_runs": int(len(group)),
                    "metric": metric,
                    "mean": mean,
                    "population_sd": population_sd,
                    "mean_pm_population_sd": f"{mean:.6f} ± {population_sd:.6f}",
                }
            )
    return pd.DataFrame(rows)


def build_confusion_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for (setting, run_id, seed), group in frame.groupby(
        ["setting", "run_id", "seed"], sort=True
    ):
        matrix = pd.crosstab(group["gold_action"], group["pred_action"]).reindex(
            index=ACTIONS, columns=PREDICTIONS, fill_value=0
        )
        for gold_action in ACTIONS:
            denominator = int(matrix.loc[gold_action].sum())
            base = {
                "setting": setting,
                "run_id": run_id,
                "seed": int(seed),
                "gold_action": gold_action,
            }
            count_rows.append(
                {
                    **base,
                    **{
                        pred_action: int(matrix.loc[gold_action, pred_action])
                        for pred_action in PREDICTIONS
                    },
                }
            )
            normalized_rows.append(
                {
                    **base,
                    **{
                        pred_action: _safe_rate(
                            int(matrix.loc[gold_action, pred_action]),
                            denominator,
                            f"confusion {gold_action}",
                        )
                        for pred_action in PREDICTIONS
                    },
                }
            )
    return pd.DataFrame(count_rows), pd.DataFrame(normalized_rows)


def build_action_recall_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting, group in per_run.groupby("setting", sort=True):
        for action in ACTIONS:
            values = group[f"recall_{action}"].dropna().astype(float).to_numpy()
            rows.append(
                {
                    "setting": setting,
                    "action": action,
                    "n_runs": int(len(values)),
                    "mean_recall": float(values.mean()) if len(values) else math.nan,
                    "population_sd": float(values.std(ddof=0))
                    if len(values)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_error_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for (setting, run_id, seed), group in frame.groupby(
        ["setting", "run_id", "seed"], sort=True
    ):
        counts = group["outcome"].value_counts()
        for outcome in OUTCOME_ORDER:
            count = int(counts.get(outcome, 0))
            base = {
                "setting": setting,
                "run_id": run_id,
                "seed": int(seed),
                "outcome": outcome,
            }
            count_rows.append({**base, "count": count})
            rate_rows.append({**base, "rate": count / len(group)})
        for action in ACTIONS:
            action_group = group[group["gold_action"] == action]
            action_counts = action_group["outcome"].value_counts()
            for outcome in OUTCOME_ORDER:
                class_rows.append(
                    {
                        "setting": setting,
                        "run_id": run_id,
                        "seed": int(seed),
                        "gold_action": action,
                        "outcome": outcome,
                        "count": int(action_counts.get(outcome, 0)),
                        "rate_within_gold": _safe_rate(
                            int(action_counts.get(outcome, 0)),
                            len(action_group),
                            f"{action}/{outcome}",
                        ),
                    }
                )
    return pd.DataFrame(count_rows), pd.DataFrame(rate_rows), pd.DataFrame(class_rows)


def paired_bootstrap_comparisons(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 20260722,
) -> pd.DataFrame:
    """Paired two-way (task ID and seed) bootstrap for every setting pair.

    Deltas are always ``setting_b - setting_a``.  Both task IDs and seeds are
    resampled as clusters, preserving pairing between settings.
    """

    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    settings = sorted(frame["setting"].unique())
    if len(settings) < 2:
        return pd.DataFrame(
            columns=[
                "setting_a",
                "setting_b",
                "metric",
                "delta_b_minus_a",
                "ci95_low",
                "ci95_high",
                "n_task_ids",
                "n_seeds",
                "n_bootstrap",
                "bootstrap_seed",
            ]
        )

    reference = frame[frame["setting"] == settings[0]].sort_values(
        ["seed", "id"], kind="stable"
    )
    ids = sorted(reference["id"].unique())
    seeds = sorted(int(seed) for seed in reference["seed"].unique())
    complete_index = pd.MultiIndex.from_product([seeds, ids], names=["seed", "id"])
    if len(reference) != len(complete_index):
        raise ValueError(f"Setting {settings[0]!r} is not a complete seed x ID panel")

    # Shape: setting x metric x seed x task.  Shared bootstrap draws preserve
    # exact pairing for every comparison and avoid redoing the expensive
    # resampling separately for all O(n^2) setting pairs.
    matrices = np.empty((len(settings), 3, len(seeds), len(ids)), dtype=np.float64)
    reference_gold: np.ndarray | None = None
    for setting_index, setting in enumerate(settings):
        indexed = frame[frame["setting"] == setting].set_index(["seed", "id"])
        if not indexed.index.is_unique or set(indexed.index) != set(complete_index):
            raise ValueError(
                f"Setting {setting!r} does not have the same complete (seed, id) panel"
            )
        indexed = indexed.reindex(complete_index)
        gold = indexed["gold_action"].to_numpy(dtype=object)
        if reference_gold is None:
            reference_gold = gold
        elif not np.array_equal(gold, reference_gold):
            raise ValueError(
                f"Setting {setting!r} disagrees on paired gold_action labels"
            )
        matrices[setting_index, 0] = (
            indexed["final_correct"]
            .astype(float)
            .to_numpy()
            .reshape(len(seeds), len(ids))
        )
        matrices[setting_index, 1] = (
            (indexed["gold_action"] == indexed["pred_action"])
            .astype(float)
            .to_numpy()
            .reshape(len(seeds), len(ids))
        )
        matrices[setting_index, 2] = (
            indexed["tool_calls"].astype(float).to_numpy().reshape(len(seeds), len(ids))
        )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_core = np.empty((n_bootstrap, len(settings), 3), dtype=np.float64)
    batch_size = 128
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        batch = stop - start
        seed_draws = rng.integers(0, len(seeds), size=(batch, len(seeds)))
        id_draws = rng.integers(0, len(ids), size=(batch, len(ids)))
        seed_weights = np.zeros((batch, len(seeds)), dtype=np.float64)
        id_weights = np.zeros((batch, len(ids)), dtype=np.float64)
        batch_rows_seed = np.repeat(np.arange(batch), len(seeds))
        batch_rows_id = np.repeat(np.arange(batch), len(ids))
        np.add.at(
            seed_weights,
            (batch_rows_seed, seed_draws.ravel()),
            1.0 / len(seeds),
        )
        np.add.at(
            id_weights,
            (batch_rows_id, id_draws.ravel()),
            1.0 / len(ids),
        )
        seed_averages = np.einsum(
            "br,smri->bsmi", seed_weights, matrices, optimize=True
        )
        bootstrap_core[start:stop] = np.einsum(
            "bsmi,bi->bsm", seed_averages, id_weights, optimize=True
        )

    point_core = matrices.mean(axis=(2, 3))
    metric_index = {
        "final_accuracy": 0,
        "action_accuracy": 1,
        "avg_tool_calls": 2,
        "total_tool_calls_per_run": 2,
    }
    rows: list[dict[str, Any]] = []
    for index_a, index_b in itertools.combinations(range(len(settings)), 2):
        setting_a = settings[index_a]
        setting_b = settings[index_b]
        for metric in BOOTSTRAP_METRICS:
            core_index = metric_index[metric]
            scale = len(ids) if metric == "total_tool_calls_per_run" else 1.0
            point_delta = (
                point_core[index_b, core_index] - point_core[index_a, core_index]
            ) * scale
            bootstrap_delta = (
                bootstrap_core[:, index_b, core_index]
                - bootstrap_core[:, index_a, core_index]
            ) * scale
            low, high = np.quantile(bootstrap_delta, [0.025, 0.975])
            rows.append(
                {
                    "setting_a": setting_a,
                    "setting_b": setting_b,
                    "metric": metric,
                    "delta_b_minus_a": float(point_delta),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "n_task_ids": len(ids),
                    "n_seeds": len(seeds),
                    "n_bootstrap": n_bootstrap,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    return pd.DataFrame(rows)


def load_label_distribution(path: Path | str) -> pd.DataFrame:
    """Load strict label rows and tabulate difficulty x category x necessity."""

    label_path = Path(path)
    payload = _read_json(label_path)
    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, Mapping) and "rows" in payload:
        raw_rows = payload["rows"]
    else:
        raise ValueError(
            f"{label_path}: labels must be a row list or an object with 'rows'"
        )
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{label_path}: label rows must be a non-empty list")

    rows: list[dict[str, Any]] = []
    split_presence = []
    id_presence = []
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, f"{label_path} rows[{index}]")
        required = {"difficulty", "category", "tool_necessary"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{label_path} rows[{index}] is missing {missing}")
        difficulty = _require_nonempty_string(
            row["difficulty"], f"{label_path} rows[{index}].difficulty"
        )
        category = row["category"]
        if category not in TOOL_ACTIONS:
            raise ValueError(
                f"{label_path} rows[{index}].category must be one of {TOOL_ACTIONS}"
            )
        necessary = row["tool_necessary"]
        if isinstance(necessary, bool):
            necessary_int = int(necessary)
        elif (
            isinstance(necessary, int)
            and not isinstance(necessary, bool)
            and necessary in (0, 1)
        ):
            necessary_int = necessary
        else:
            raise TypeError(
                f"{label_path} rows[{index}].tool_necessary must be boolean or 0/1"
            )
        if "gold_action" in row:
            expected = category if necessary_int else "NONE"
            if row["gold_action"] != expected:
                raise ValueError(
                    f"{label_path} rows[{index}].gold_action={row['gold_action']!r}; expected {expected!r}"
                )
        result = {
            "difficulty": difficulty,
            "category": category,
            "tool_necessary": necessary_int,
        }
        split_presence.append("split" in row)
        if "split" in row:
            result["split"] = _require_nonempty_string(
                row["split"], f"{label_path} rows[{index}].split"
            )
        id_presence.append("id" in row)
        if "id" in row:
            result["id"] = _canonical_identifier(
                row["id"], f"{label_path} rows[{index}].id"
            )
        rows.append(result)
    if any(split_presence) and not all(split_presence):
        raise ValueError(
            f"{label_path}: split must be present on every label row or none"
        )
    if any(id_presence) and not all(id_presence):
        raise ValueError(f"{label_path}: id must be present on every label row or none")

    frame = pd.DataFrame(rows)
    if all(id_presence):
        duplicate_keys = ["id"] if not all(split_presence) else ["split", "id"]
        if frame.duplicated(duplicate_keys).any():
            raise ValueError(
                f"{label_path}: duplicate label IDs for key {duplicate_keys}"
            )
    group_prefix = ["split"] if all(split_presence) else []
    group_columns = group_prefix + ["difficulty", "category", "tool_necessary"]
    counts = (
        frame.groupby(group_columns, sort=True).size().rename("count").reset_index()
    )
    cell_columns = group_prefix + ["difficulty", "category"]
    totals = counts.groupby(cell_columns, sort=True)["count"].transform("sum")
    counts["cell_total"] = totals
    counts["proportion_within_difficulty_category"] = counts["count"] / totals
    return counts


def _plot_confusions(frame: pd.DataFrame, path: Path) -> None:
    settings = sorted(frame["setting"].unique())
    figure, axes = plt.subplots(
        1,
        len(settings),
        figsize=(6.0 * len(settings), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, setting in zip(axes.flat, settings):
        subset = frame[frame["setting"] == setting]
        matrix = pd.crosstab(subset["gold_action"], subset["pred_action"]).reindex(
            index=ACTIONS, columns=PREDICTIONS, fill_value=0
        )
        normalized = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(
            0.0
        )
        sns.heatmap(
            normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            square=False,
            cbar=axis is axes.flat[-1],
            ax=axis,
        )
        axis.set_title(setting)
        axis.set_xlabel("Predicted first action")
        axis.set_ylabel("Gold action")
    figure.suptitle("Action confusion matrices (row-normalized)", fontsize=13)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_recalls(recall_summary: pd.DataFrame, path: Path) -> None:
    settings = sorted(recall_summary["setting"].unique())
    x = np.arange(len(settings), dtype=float)
    width = 0.18
    colors = sns.color_palette("colorblind", n_colors=len(ACTIONS))
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(settings) * 1.8), 4.8), constrained_layout=True
    )
    for index, (action, color) in enumerate(zip(ACTIONS, colors)):
        subset = (
            recall_summary[recall_summary["action"] == action]
            .set_index("setting")
            .reindex(settings)
        )
        offsets = x + (index - (len(ACTIONS) - 1) / 2) * width
        axis.bar(
            offsets,
            subset["mean_recall"],
            width,
            yerr=subset["population_sd"],
            capsize=3,
            label=action,
            color=color,
        )
    axis.set_xticks(x, settings, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Recall (mean ± population SD)")
    axis.set_title("Per-action recall across runs")
    axis.legend(title="Gold action", ncol=len(ACTIONS))
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_error_stack(error_rates: pd.DataFrame, path: Path) -> None:
    mean_rates = (
        error_rates.groupby(["setting", "outcome"], sort=True)["rate"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0.0)
    )
    colors = sns.color_palette("Set2", n_colors=len(OUTCOME_ORDER))
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(mean_rates) * 1.7), 4.8), constrained_layout=True
    )
    bottom = np.zeros(len(mean_rates), dtype=float)
    x = np.arange(len(mean_rates))
    for outcome, color in zip(OUTCOME_ORDER, colors):
        values = mean_rates[outcome].to_numpy(dtype=float)
        axis.bar(x, values, bottom=bottom, label=outcome, color=color)
        bottom += values
    axis.set_xticks(x, mean_rates.index, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean fraction of tasks")
    axis.set_title("Mutually exclusive outcome hierarchy")
    axis.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_accuracy_vs_calls(per_run: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    sns.scatterplot(
        data=per_run,
        x="total_tool_calls",
        y="final_accuracy",
        hue="setting",
        style="setting",
        s=75,
        ax=axis,
    )
    means = per_run.groupby("setting", sort=True)[
        ["total_tool_calls", "final_accuracy"]
    ].mean()
    axis.scatter(
        means["total_tool_calls"],
        means["final_accuracy"],
        marker="X",
        s=150,
        c="black",
        label="setting mean",
        zorder=5,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Total executed tool calls per run")
    axis.set_ylabel("Final-answer accuracy")
    axis.set_title("Accuracy–tool-call trade-off")
    axis.grid(alpha=0.25)
    axis.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _summary_payload(
    frame: pd.DataFrame,
    per_run: pd.DataFrame,
    run_summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_paths: Sequence[Path],
    labels_path: Path | None,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for setting, group in run_summary.groupby("setting", sort=True):
        settings[setting] = {
            "n_runs": int(group["n_runs"].iloc[0]),
            "metrics": {
                row.metric: {
                    "mean": _json_scalar(row.mean),
                    "population_sd": _json_scalar(row.population_sd),
                }
                for row in group.itertuples(index=False)
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "input_files": [str(path) for path in output_paths],
        "labels_file": None if labels_path is None else str(labels_path),
        "n_rows": int(len(frame)),
        "n_task_ids": int(frame["id"].nunique()),
        "n_settings": int(frame["setting"].nunique()),
        "n_runs": int(len(per_run)),
        "actions": list(ACTIONS),
        "prediction_columns": list(PREDICTIONS),
        "outcome_plot_order": list(OUTCOME_ORDER),
        "error_hierarchy": list(ERROR_HIERARCHY),
        "metric_definitions": METRIC_DEFINITIONS,
        "aggregation": "Per-run metrics followed by arithmetic mean and population SD (ddof=0).",
        "paired_bootstrap": {
            "method": "Two-way paired cluster bootstrap over task IDs and seeds; deltas are setting_b - setting_a.",
            "n_bootstrap": n_bootstrap,
            "seed": bootstrap_seed,
            "comparisons": [
                {key: _json_scalar(value) for key, value in row.items()}
                for row in paired.to_dict(orient="records")
            ],
        },
        "settings": settings,
    }


def _prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists and is not a directory: {output_dir}"
            )
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}; pass --overwrite intentionally"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def collect_action_statistics(
    output_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    labels_path: Path | str | None = None,
    overwrite: bool = False,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 20260722,
) -> dict[str, Any]:
    """Validate evaluation outputs and write all statistics and plots."""

    normalized_inputs = [Path(path) for path in output_paths]
    destination = Path(output_dir)
    normalized_labels = None if labels_path is None else Path(labels_path)
    if destination in normalized_inputs or normalized_labels == destination:
        raise ValueError("Output directory must differ from every input path")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(
            f"Output path exists and is not a directory: {destination}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {destination}; pass --overwrite intentionally"
        )

    frame = load_evaluation_outputs(normalized_inputs)
    per_run = compute_per_run_metrics(frame)
    run_summary = summarize_run_metrics(per_run)
    confusion_counts, confusion_rates = build_confusion_tables(frame)
    recall_summary = build_action_recall_summary(per_run)
    error_counts, error_rates, class_error_rates = build_error_tables(frame)
    paired = paired_bootstrap_comparisons(
        frame, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
    )
    label_distribution = (
        None
        if normalized_labels is None
        else load_label_distribution(normalized_labels)
    )
    _prepare_output_directory(destination, overwrite)

    tables = {
        "derived_action_rows.csv": frame,
        "per_run_metrics.csv": per_run,
        "setting_metric_summary.csv": run_summary,
        "confusion_counts.csv": confusion_counts,
        "confusion_row_normalized.csv": confusion_rates,
        "action_recall_summary.csv": recall_summary,
        "outcome_counts.csv": error_counts,
        "outcome_rates.csv": error_rates,
        "class_outcome_rates.csv": class_error_rates,
        "paired_bootstrap_comparisons.csv": paired,
    }
    if label_distribution is not None:
        tables["label_distribution.csv"] = label_distribution
    for filename, table in tables.items():
        table.to_csv(destination / filename, index=False)

    _plot_confusions(frame, destination / "confusion_heatmap.png")
    _plot_recalls(recall_summary, destination / "recall_bars.png")
    _plot_error_stack(error_rates, destination / "error_stacked.png")
    _plot_accuracy_vs_calls(per_run, destination / "accuracy_vs_total_tc.png")

    summary = _summary_payload(
        frame,
        per_run,
        run_summary,
        paired,
        normalized_inputs,
        normalized_labels,
        n_bootstrap,
        bootstrap_seed,
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
