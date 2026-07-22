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

from .constants import SCHEMA_VERSION as ACTION_SCHEMA_VERSION
from .constants import UPSTREAM_COMMIT
from .evaluation_contract import validate_action_row
from .io_utils import sha256_file


ACTIONS = ("NONE", "A", "B", "C")
PREDICTIONS = (*ACTIONS, "INVALID")
TOOL_ACTIONS = ("A", "B", "C")
OUTCOME_ORDER = (
    "success",
    "direct_answer_wrong",
    "correct_category_wrong_answer",
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
    "direct_answer_wrong/correct_category_wrong_answer (action is correct but final answer is wrong)",
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
    "balanced_accuracy",
    "macro_f1",
    "toolneed_f1",
    "recall_NONE",
    "recall_A",
    "recall_B",
    "recall_C",
    "overcall_rate",
)
SCHEMA_VERSION = "when2tool_action_stats.v3"

METRIC_DEFINITIONS = {
    "final_accuracy": "Fraction of tasks whose final answer is correct.",
    "total_tool_calls": "Sum of executed/routed tool calls in the run.",
    "avg_tool_calls": "Mean executed/routed tool calls per task.",
    "tool_call_rate": (
        "Tool-call rate (TCR) for this single-hop benchmark: total executed/routed "
        "calls divided by the number of tasks. This is a call-volume metric, not "
        "the fraction of tasks that called, and can exceed 1 when multi-calls occur."
    ),
    "action_accuracy": "Accuracy of the first executed tool category, with zero calls mapped to NONE.",
    "balanced_accuracy": "Unweighted mean recall over gold action classes present in the run.",
    "macro_f1": "Macro F1 over gold classes NONE/A/B/C; INVALID predictions count as misses.",
    "toolneed_f1": "Binary F1 for tool-needed (A/B/C) versus NONE; INVALID is a tool attempt.",
    "overcall_rate": "Among gold NONE tasks, fraction with one or more calls.",
    "no_call_precision": "P(gold NONE | predicted NONE).",
    "no_call_recall": "P(predicted NONE | gold NONE); equivalently recall_NONE.",
    "under_call_rate": "Among gold A/B/C tasks, fraction with zero calls.",
    "category_accuracy_needed_given_call": (
        "Among gold A/B/C tasks whose first prediction is a valid called category "
        "A/B/C, fraction whose predicted category equals the gold category."
    ),
    "wrong_category_rate": "Among gold A/B/C tasks, fraction whose first call is another valid category.",
    "invalid_tool_rate": "Fraction of tasks with an INVALID category anywhere in the routed call sequence.",
    "mixed_call_rate": "Fraction of all tasks whose routed call sequence contains more than one category.",
    "multi_call_rate": "Fraction of tasks with more than one executed/routed tool call.",
    "mixed_over_multicall_rate": (
        "Among tasks with more than one call, fraction whose routed category sequence "
        "contains more than one distinct category."
    ),
    "majority_accuracy_baseline": "Accuracy of always predicting the most frequent gold action in that run.",
    "prior_matched_expected_accuracy": "Expected accuracy of sampling predictions from the empirical gold prior, sum_c p(c)^2.",
}

TABLE_DEFINITIONS = {
    "needed_category_analysis.csv": (
        "One row per setting/run/gold A/B/C. call_correct_category is first "
        "predicted action == gold; under_call is predicted NONE; wrong_category "
        "is another valid A/B/C first action; correct_category_wrong_answer and "
        "needed_final_success additionally split by final_correct. invalid_tool is "
        "reported separately and can overlap when a later routed event is invalid."
    ),
    "none_analysis.csv": (
        "One row per setting/run for gold NONE. Over-call A/B/C/INVALID uses the "
        "first predicted action and therefore partitions over_call_total; "
        "any_invalid_tool separately detects INVALID anywhere in the sequence."
    ),
    "multicall_summary.csv": (
        "One row per setting/run with call-count rates. Top category sequences are "
        "the ten most frequent mixed-category paths only, sorted by count then "
        "lexicographically; the first three also have dedicated CSV columns."
    ),
    "multicall_by_gold.csv": (
        "The multicall summary split by gold NONE/A/B/C, including top "
        "mixed-category sequence counts."
    ),
    "run_diagnostics.csv": (
        "One row per setting/run with mutually exclusive termination counts, "
        "tool-parse failures, and explicit [SAFETY_REJECTED] tool results."
    ),
    "gold_action_final_accuracy_summary.csv": (
        "Final-answer accuracy grouped by setting and gold action, aggregated "
        "over complete runs as mean and population SD."
    ),
    "current_relative_tradeoff_summary.csv": (
        "Per-setting mean and population SD for tool-call reduction, accuracy "
        "loss, and accuracy loss per saved call relative to the matching current "
        "setting and seed."
    ),
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


def _contains_safety_rejection(value: Any) -> bool:
    """Detect the explicit runtime marker without treating every failure as safety."""

    if isinstance(value, str):
        return "[SAFETY_REJECTED]" in value
    if isinstance(value, Mapping):
        return any(_contains_safety_rejection(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_safety_rejection(item) for item in value)
    return False


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
        "setting",
        "termination_reason",
        "tool_parse_failures",
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
    contract = validate_action_row(
        row,
        context=context,
        require_pred_action=False,
        strict_event_categories=False,
    )
    gold_action = contract.gold_action
    final_correct = contract.final_correct
    tool_calls = contract.tool_calls
    raw_events = row["routed_tool_events"]
    events = contract.events
    category_sequence = list(contract.category_sequence)
    pred_action = contract.pred_action
    has_invalid_tool = contract.invalid_tool_calls > 0
    unique_categories = list(contract.unique_categories)
    mixed_calls = contract.mixed_category_calls
    outcome = contract.outcome
    termination_reason = _require_nonempty_string(
        row["termination_reason"], f"{context}.termination_reason"
    )
    if termination_reason not in {"boxed_answer", "max_rounds"}:
        raise ValueError(
            f"{context}.termination_reason must be boxed_answer or max_rounds, "
            f"got {termination_reason!r}"
        )
    tool_parse_failures = _require_int(
        row["tool_parse_failures"], f"{context}.tool_parse_failures", minimum=0
    )
    safety_rejections = 0
    for index, event in enumerate(events):
        if "result" not in event:
            raise ValueError(
                f"{context}.routed_tool_events[{index}] is missing 'result'; "
                "formal safety diagnostics require the persisted tool result"
            )
        event_result = event["result"]
        if not isinstance(event_result, Mapping):
            raise TypeError(
                f"{context}.routed_tool_events[{index}].result must be an object"
            )
        safety_rejections += int(_contains_safety_rejection(event_result))

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
        "tool_call_categories": json.dumps(category_sequence, ensure_ascii=False),
        "unique_tool_call_categories": json.dumps(
            unique_categories, ensure_ascii=False
        ),
        "n_tool_call_categories": len(unique_categories),
        "category_sequence_text": ">".join(category_sequence)
        if category_sequence
        else "NONE",
        "mixed_calls": mixed_calls,
        "mixed_category_calls": mixed_calls,
        "has_invalid_tool": has_invalid_tool,
        "termination_reason": termination_reason,
        "tool_parse_failures": tool_parse_failures,
        "safety_rejection_count": safety_rejections,
        "has_safety_rejection": safety_rejections > 0,
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


def validate_expected_seed_panel(
    frame: pd.DataFrame, expected_seeds: Sequence[int]
) -> tuple[int, ...]:
    """Require an exact seed set for every experimental setting.

    Callers opt in to this formal-panel check by passing ``expected_seeds``.
    Intentionally partial smoke analyses must omit that argument explicitly.
    """

    normalized: list[int] = []
    for index, seed in enumerate(expected_seeds):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(
                f"expected_seeds[{index}] must be an integer, got {seed!r}"
            )
        normalized.append(int(seed))
    if not normalized:
        raise ValueError("expected_seeds must be non-empty when provided")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"expected_seeds must be unique, got {normalized}")

    expected = set(normalized)
    for setting, group in frame.groupby("setting", sort=True):
        actual = {int(seed) for seed in group["seed"].unique()}
        if actual != expected:
            raise ValueError(
                f"Setting {setting!r} does not have the exact expected seed panel; "
                f"expected={sorted(expected)}, actual={sorted(actual)}, "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
    return tuple(normalized)


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
    pred_none = ~pred_need
    needed_with_valid_call = gold_need & np.isin(pred, TOOL_ACTIONS)
    multi_call = group["tool_calls"].to_numpy(dtype=int) > 1
    total_tool_calls = int(group["tool_calls"].sum())
    tool_call_rate = float(total_tool_calls / n_tasks)
    prior = pd.Series(gold).value_counts(normalize=True)
    row: dict[str, Any] = {
        "setting": setting_values[0],
        "run_id": run_values[0],
        "seed": int(seed_values[0]),
        "n_tasks": n_tasks,
        "final_accuracy": float(group["final_correct"].mean()),
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": float(group["tool_calls"].mean()),
        "tool_call_rate": tool_call_rate,
        "action_accuracy": float(np.mean(gold == pred)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(
            f1_score(gold, pred, labels=list(ACTIONS), average="macro", zero_division=0)
        ),
        "toolneed_f1": float(f1_score(gold_need, pred_need, zero_division=0)),
        "overcall_rate": _safe_rate(
            np.sum(pred[gold_none] != "NONE"), int(gold_none.sum()), "overcall"
        ),
        "no_call_precision": _safe_rate(
            np.sum(gold[pred_none] == "NONE"), int(pred_none.sum()), "no-call precision"
        ),
        "no_call_recall": _safe_rate(
            np.sum(pred[gold_none] == "NONE"), int(gold_none.sum()), "no-call recall"
        ),
        "under_call_rate": _safe_rate(
            np.sum(pred[gold_need] == "NONE"), int(gold_need.sum()), "under-call"
        ),
        "category_accuracy_needed_given_call": _safe_rate(
            np.sum(pred[needed_with_valid_call] == gold[needed_with_valid_call]),
            int(needed_with_valid_call.sum()),
            "needed category accuracy given a valid call",
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
        "multi_call_rate": float(multi_call.mean()),
        "mixed_over_multicall_rate": _safe_rate(
            np.sum(group["mixed_calls"].to_numpy(dtype=bool) & multi_call),
            int(multi_call.sum()),
            "mixed over multi-call",
        ),
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
                    "n_runs": int(len(values)),
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


def build_needed_category_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Break down routing and answer outcomes for gold A/B/C tasks per run."""

    rows: list[dict[str, Any]] = []
    for (setting, run_id, seed), run_group in frame.groupby(
        ["setting", "run_id", "seed"], sort=True
    ):
        for gold_action in TOOL_ACTIONS:
            group = run_group[run_group["gold_action"] == gold_action]
            n_tasks = len(group)
            pred = group["pred_action"]
            correct_category = pred == gold_action
            under_call = pred == "NONE"
            wrong_category = pred.isin(TOOL_ACTIONS) & ~correct_category
            invalid_tool = group["has_invalid_tool"]
            correct_category_wrong_answer = correct_category & ~group["final_correct"]
            final_success = correct_category & group["final_correct"]
            row: dict[str, Any] = {
                "setting": setting,
                "run_id": run_id,
                "seed": int(seed),
                "gold_action": gold_action,
                "n_tasks": n_tasks,
            }
            for name, mask in (
                ("call_correct_category", correct_category),
                ("under_call", under_call),
                ("wrong_category", wrong_category),
                ("invalid_tool", invalid_tool),
                (
                    "correct_category_wrong_answer",
                    correct_category_wrong_answer,
                ),
                ("needed_final_success", final_success),
            ):
                count = int(mask.sum())
                row[f"{name}_count"] = count
                row[f"{name}_rate"] = _safe_rate(
                    count, n_tasks, f"needed category {gold_action}/{name}"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_none_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Break down direct answers and first-action over-calls on gold NONE tasks."""

    rows: list[dict[str, Any]] = []
    for (setting, run_id, seed), run_group in frame.groupby(
        ["setting", "run_id", "seed"], sort=True
    ):
        group = run_group[run_group["gold_action"] == "NONE"]
        n_tasks = len(group)
        pred = group["pred_action"]
        masks: dict[str, pd.Series] = {
            "no_call_correct": (pred == "NONE") & group["final_correct"],
            "direct_answer_wrong": group["outcome"] == "direct_answer_wrong",
            "overcall_A": pred == "A",
            "overcall_B": pred == "B",
            "overcall_C": pred == "C",
            "overcall_INVALID": pred == "INVALID",
            "overcall_total": pred != "NONE",
            "any_invalid_tool": group["has_invalid_tool"],
        }
        row: dict[str, Any] = {
            "setting": setting,
            "run_id": run_id,
            "seed": int(seed),
            "n_tasks": n_tasks,
        }
        for name, mask in masks.items():
            count = int(mask.sum())
            row[f"{name}_count"] = count
            row[f"{name}_rate"] = _safe_rate(count, n_tasks, f"NONE analysis/{name}")
        rows.append(row)
    return pd.DataFrame(rows)


def _multicall_group_summary(group: pd.DataFrame) -> dict[str, Any]:
    n_tasks = len(group)
    tool_calls = group["tool_calls"].to_numpy(dtype=int)
    zero_call = tool_calls == 0
    one_call = tool_calls == 1
    any_call = tool_calls > 0
    multi_call = tool_calls > 1
    mixed_multicall = multi_call & group["mixed_calls"].to_numpy(dtype=bool)
    mixed_sequences = group.loc[mixed_multicall, "category_sequence_text"]
    sequence_counts = sorted(
        (
            (str(sequence).replace(">", "->"), int(count))
            for sequence, count in mixed_sequences.value_counts().items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    top_sequences = [
        {
            "rank": rank,
            "category_sequence": sequence,
            "count": count,
            "rate_within_multicall": _safe_rate(
                count, int(multi_call.sum()), f"top multi-call sequence rank {rank}"
            ),
            "rate_within_mixed": _safe_rate(
                count, int(mixed_multicall.sum()), f"top mixed sequence rank {rank}"
            ),
        }
        for rank, (sequence, count) in enumerate(sequence_counts[:10], start=1)
    ]
    result: dict[str, Any] = {
        "n_tasks": n_tasks,
        "total_tool_calls": int(tool_calls.sum()),
        "tool_call_rate": math.nan
        if n_tasks == 0
        else float(tool_calls.sum() / n_tasks),
        "zero_call_count": int(zero_call.sum()),
        "zero_call_rate": _safe_rate(int(zero_call.sum()), n_tasks, "zero-call rate"),
        "any_tool_call_count": int(any_call.sum()),
        "any_tool_call_rate": _safe_rate(
            int(any_call.sum()), n_tasks, "any-tool-call rate"
        ),
        "one_call_count": int(one_call.sum()),
        "one_call_rate": _safe_rate(int(one_call.sum()), n_tasks, "one-call rate"),
        "multi_call_count": int(multi_call.sum()),
        "multi_call_rate": _safe_rate(
            int(multi_call.sum()), n_tasks, "multi-call rate"
        ),
        "mixed_multicall_count": int(mixed_multicall.sum()),
        "mixed_category_task_count": int(mixed_multicall.sum()),
        "mixed_over_all_rate": _safe_rate(
            int(mixed_multicall.sum()), n_tasks, "mixed over all tasks"
        ),
        "mixed_over_multicall_rate": _safe_rate(
            int(mixed_multicall.sum()),
            int(multi_call.sum()),
            "mixed over multi-call",
        ),
        "top_category_sequences_json": json.dumps(
            top_sequences, ensure_ascii=False, separators=(",", ":")
        ),
    }
    for rank in range(1, 4):
        if rank <= len(top_sequences):
            top = top_sequences[rank - 1]
            result[f"top_{rank}_category_sequence"] = top["category_sequence"]
            result[f"top_{rank}_sequence_count"] = top["count"]
            result[f"top_{rank}_sequence_rate_within_multicall"] = top[
                "rate_within_multicall"
            ]
            result[f"top_{rank}_sequence_rate_within_mixed"] = top["rate_within_mixed"]
        else:
            result[f"top_{rank}_category_sequence"] = None
            result[f"top_{rank}_sequence_count"] = 0
            result[f"top_{rank}_sequence_rate_within_multicall"] = math.nan
            result[f"top_{rank}_sequence_rate_within_mixed"] = math.nan
    return result


def build_multicall_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize call multiplicity and the ten most common multi-call sequences."""

    summary_rows: list[dict[str, Any]] = []
    by_gold_rows: list[dict[str, Any]] = []
    for (setting, run_id, seed), group in frame.groupby(
        ["setting", "run_id", "seed"], sort=True
    ):
        base = {"setting": setting, "run_id": run_id, "seed": int(seed)}
        summary_rows.append({**base, **_multicall_group_summary(group)})
        for gold_action in ACTIONS:
            action_group = group[group["gold_action"] == gold_action]
            by_gold_rows.append(
                {
                    **base,
                    "gold_action": gold_action,
                    **_multicall_group_summary(action_group),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(by_gold_rows)


def _tidy_group_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_columns, sort=True, dropna=False)
    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        assert len(group_columns) == len(key_values)
        base = dict(zip(group_columns, key_values))
        for metric in metric_columns:
            values = group[metric].dropna().astype(float).to_numpy()
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_runs": int(len(values)),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "population_sd": float(values.std(ddof=0))
                    if len(values)
                    else math.nan,
                    "mean_pm_population_sd": (
                        f"{values.mean():.6f} ± {values.std(ddof=0):.6f}"
                        if len(values)
                        else "undefined"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_run_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize formal termination, parse, and safety diagnostics per run."""

    rows: list[dict[str, Any]] = []
    for (setting, run_id, seed), group in frame.groupby(
        ["setting", "run_id", "seed"], sort=True
    ):
        n_tasks = len(group)
        boxed = group["termination_reason"] == "boxed_answer"
        max_rounds = group["termination_reason"] == "max_rounds"
        if int(boxed.sum() + max_rounds.sum()) != n_tasks:
            raise AssertionError("Termination reasons do not partition a run")
        parse_failures = group["tool_parse_failures"].to_numpy(dtype=int)
        safety_rejections = group["safety_rejection_count"].to_numpy(dtype=int)
        rows.append(
            {
                "setting": setting,
                "run_id": run_id,
                "seed": int(seed),
                "n_tasks": n_tasks,
                "boxed_answer_count": int(boxed.sum()),
                "boxed_answer_rate": float(boxed.mean()),
                "max_rounds_count": int(max_rounds.sum()),
                "max_rounds_rate": float(max_rounds.mean()),
                "rows_with_parse_failures_count": int((parse_failures > 0).sum()),
                "rows_with_parse_failures_rate": float((parse_failures > 0).mean()),
                "total_tool_parse_failures": int(parse_failures.sum()),
                "avg_tool_parse_failures": float(parse_failures.mean()),
                "rows_with_safety_rejection_count": int(
                    (safety_rejections > 0).sum()
                ),
                "rows_with_safety_rejection_rate": float(
                    (safety_rejections > 0).mean()
                ),
                "total_safety_rejections": int(safety_rejections.sum()),
                "avg_safety_rejections": float(safety_rejections.mean()),
            }
        )
    per_run = pd.DataFrame(rows)
    metrics = [
        column
        for column in per_run.columns
        if column not in {"setting", "run_id", "seed"}
    ]
    return per_run, _tidy_group_summary(
        per_run, group_columns=["setting"], metric_columns=metrics
    )


def build_gold_action_final_accuracy(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-run and across-run final accuracy for each gold action."""

    rows: list[dict[str, Any]] = []
    for (setting, run_id, seed, gold_action), group in frame.groupby(
        ["setting", "run_id", "seed", "gold_action"], sort=True
    ):
        rows.append(
            {
                "setting": setting,
                "run_id": run_id,
                "seed": int(seed),
                "gold_action": gold_action,
                "n_tasks": len(group),
                "final_correct_count": int(group["final_correct"].sum()),
                "final_accuracy": float(group["final_correct"].mean()),
            }
        )
    per_run = pd.DataFrame(rows)
    summary = _tidy_group_summary(
        per_run,
        group_columns=["setting", "gold_action"],
        metric_columns=["final_accuracy"],
    )
    return per_run, summary


def _setting_comparison_group(setting: str) -> tuple[str, str]:
    if "no_reasoning" in setting or setting.startswith("probe_prefill_"):
        reasoning = "no_reasoning"
    elif "reasoning" in setting:
        reasoning = "reasoning"
    else:
        reasoning = "unspecified"
    if "fulltools" in setting:
        scope = "fulltools"
    elif "scoped" in setting:
        scope = "scoped"
    else:
        scope = "unspecified"
    return reasoning, scope


def build_current_relative_tradeoff(
    per_run: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair every run to the matching current setting at the same seed."""

    settings = sorted(str(value) for value in per_run["setting"].unique())
    references: dict[tuple[str, str], str] = {}
    for setting in settings:
        if setting == "current" or setting.startswith("current_"):
            group = _setting_comparison_group(setting)
            if group in references:
                raise ValueError(
                    f"Multiple current references for comparison group {group}: "
                    f"{references[group]!r}, {setting!r}"
                )
            references[group] = setting
    rows: list[dict[str, Any]] = []
    for setting in settings:
        group = _setting_comparison_group(setting)
        reference_setting = references.get(group)
        if reference_setting is None:
            raise ValueError(
                f"No current reference setting for {setting!r} in group {group}"
            )
        target = per_run[per_run["setting"] == setting].set_index("seed")
        reference = per_run[
            per_run["setting"] == reference_setting
        ].set_index("seed")
        if not target.index.is_unique or not reference.index.is_unique:
            raise ValueError("Current-relative comparison requires one run per setting/seed")
        if set(target.index) != set(reference.index):
            raise ValueError(
                f"Seed panel differs for {setting!r} and {reference_setting!r}"
            )
        for seed in sorted(target.index):
            target_row = target.loc[seed]
            reference_row = reference.loc[seed]
            current_calls = float(reference_row["total_tool_calls"])
            setting_calls = float(target_row["total_tool_calls"])
            calls_saved = current_calls - setting_calls
            accuracy_loss = float(
                reference_row["final_accuracy"] - target_row["final_accuracy"]
            )
            tc_reduction = (
                calls_saved / current_calls if current_calls > 0 else math.nan
            )
            cost_per_saved_call = (
                accuracy_loss / calls_saved if calls_saved > 0 else math.nan
            )
            rows.append(
                {
                    "setting": setting,
                    "reference_setting": reference_setting,
                    "comparison_group": "/".join(group),
                    "seed": int(seed),
                    "current_total_tool_calls": current_calls,
                    "setting_total_tool_calls": setting_calls,
                    "tool_calls_saved": calls_saved,
                    "tc_reduction": tc_reduction,
                    "current_final_accuracy": float(reference_row["final_accuracy"]),
                    "setting_final_accuracy": float(target_row["final_accuracy"]),
                    "accuracy_loss": accuracy_loss,
                    "cost_per_saved_call": cost_per_saved_call,
                    "cost_per_saved_call_defined": calls_saved > 0,
                }
            )
    per_run_tradeoff = pd.DataFrame(rows)
    summary = _tidy_group_summary(
        per_run_tradeoff,
        group_columns=["setting", "reference_setting", "comparison_group"],
        metric_columns=[
            "tool_calls_saved",
            "tc_reduction",
            "accuracy_loss",
            "cost_per_saved_call",
        ],
    )
    return per_run_tradeoff, summary


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
                "n_bootstrap_valid",
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

    # Shape: setting x feature x seed x task.  The first three features are
    # linear outcomes; the remaining 20 are the fixed 4x5 gold/prediction
    # contingency cells.  Shared draws preserve exact pairing across settings.
    n_linear = 3
    n_confusion = len(ACTIONS) * len(PREDICTIONS)
    matrices = np.empty(
        (len(settings), n_linear + n_confusion, len(seeds), len(ids)),
        dtype=np.float64,
    )
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
        gold_grid = indexed["gold_action"].to_numpy(dtype=object).reshape(
            len(seeds), len(ids)
        )
        pred_grid = indexed["pred_action"].to_numpy(dtype=object).reshape(
            len(seeds), len(ids)
        )
        feature = n_linear
        for gold_action in ACTIONS:
            for pred_action in PREDICTIONS:
                matrices[setting_index, feature] = (
                    (gold_grid == gold_action) & (pred_grid == pred_action)
                ).astype(float)
                feature += 1

    assert reference_gold is not None
    reference_gold_grid = reference_gold.reshape(len(seeds), len(ids))
    for seed_index, seed in enumerate(seeds):
        missing = sorted(set(ACTIONS) - set(reference_gold_grid[seed_index]))
        if missing:
            raise ValueError(
                "Paired bootstrap metrics require every gold action in every seed; "
                f"seed {seed} is missing {missing}"
            )

    metric_index = {metric: index for index, metric in enumerate(BOOTSTRAP_METRICS)}

    def derive_metrics(task_averages: np.ndarray) -> np.ndarray:
        """Convert task-weighted features to per-seed registered metrics.

        Input shape is batch x setting x feature x seed; output shape is
        batch x setting x metric x seed.  A bootstrap draw that omits a gold
        class yields NaN for its recall, balanced accuracy, and over-call rate,
        never an implicit zero.
        """

        confusion = task_averages[:, :, n_linear:, :].reshape(
            len(task_averages),
            len(settings),
            len(ACTIONS),
            len(PREDICTIONS),
            len(seeds),
        )
        row_totals = confusion.sum(axis=3)
        true_positive = np.stack(
            [confusion[:, :, index, index, :] for index in range(len(ACTIONS))],
            axis=2,
        )
        recalls = np.full_like(true_positive, np.nan)
        np.divide(
            true_positive,
            row_totals,
            out=recalls,
            where=row_totals > 0,
        )
        balanced = recalls.mean(axis=2)

        f1_values: list[np.ndarray] = []
        for action_index in range(len(ACTIONS)):
            tp = true_positive[:, :, action_index, :]
            fp = confusion[:, :, :, action_index, :].sum(axis=2) - tp
            fn = row_totals[:, :, action_index, :] - tp
            denominator = 2.0 * tp + fp + fn
            score = np.zeros_like(tp)
            np.divide(2.0 * tp, denominator, out=score, where=denominator > 0)
            f1_values.append(score)
        macro_f1 = np.stack(f1_values, axis=2).mean(axis=2)

        tool_tp = confusion[:, :, 1:4, 1:5, :].sum(axis=(2, 3))
        tool_fp = confusion[:, :, 0, 1:5, :].sum(axis=2)
        tool_fn = confusion[:, :, 1:4, 0, :].sum(axis=2)
        tool_denominator = 2.0 * tool_tp + tool_fp + tool_fn
        toolneed_f1 = np.zeros_like(tool_tp)
        np.divide(
            2.0 * tool_tp,
            tool_denominator,
            out=toolneed_f1,
            where=tool_denominator > 0,
        )
        overcall = np.full_like(tool_fp, np.nan)
        np.divide(
            tool_fp,
            row_totals[:, :, 0, :],
            out=overcall,
            where=row_totals[:, :, 0, :] > 0,
        )

        values = np.empty(
            (
                len(task_averages),
                len(settings),
                len(BOOTSTRAP_METRICS),
                len(seeds),
            ),
            dtype=np.float64,
        )
        values[:, :, metric_index["final_accuracy"], :] = task_averages[:, :, 0, :]
        values[:, :, metric_index["action_accuracy"], :] = task_averages[:, :, 1, :]
        values[:, :, metric_index["avg_tool_calls"], :] = task_averages[:, :, 2, :]
        values[:, :, metric_index["total_tool_calls_per_run"], :] = (
            task_averages[:, :, 2, :] * len(ids)
        )
        values[:, :, metric_index["balanced_accuracy"], :] = balanced
        values[:, :, metric_index["macro_f1"], :] = macro_f1
        values[:, :, metric_index["toolneed_f1"], :] = toolneed_f1
        for action_index, action in enumerate(ACTIONS):
            values[:, :, metric_index[f"recall_{action}"], :] = recalls[
                :, :, action_index, :
            ]
        values[:, :, metric_index["overcall_rate"], :] = overcall
        return values

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_metrics = np.empty(
        (n_bootstrap, len(settings), len(BOOTSTRAP_METRICS)), dtype=np.float64
    )
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
        task_averages = np.einsum(
            "sfri,bi->bsfr", matrices, id_weights, optimize=True
        )
        per_seed_metrics = derive_metrics(task_averages)
        bootstrap_metrics[start:stop] = np.einsum(
            "br,bsmr->bsm", seed_weights, per_seed_metrics, optimize=True
        )

    uniform_task_weights = np.full((1, len(ids)), 1.0 / len(ids))
    point_task_averages = np.einsum(
        "sfri,bi->bsfr", matrices, uniform_task_weights, optimize=True
    )
    point_per_seed = derive_metrics(point_task_averages)
    point_metrics = point_per_seed.mean(axis=3)[0]
    rows: list[dict[str, Any]] = []
    for index_a, index_b in itertools.combinations(range(len(settings)), 2):
        setting_a = settings[index_a]
        setting_b = settings[index_b]
        for metric in BOOTSTRAP_METRICS:
            index = metric_index[metric]
            point_delta = (
                point_metrics[index_b, index] - point_metrics[index_a, index]
            )
            bootstrap_delta = (
                bootstrap_metrics[:, index_b, index]
                - bootstrap_metrics[:, index_a, index]
            )
            valid = np.isfinite(bootstrap_delta)
            n_valid = int(valid.sum())
            if not math.isfinite(float(point_delta)) or n_valid == 0:
                raise ValueError(
                    f"Paired bootstrap metric {metric} is undefined for the formal panel"
                )
            low, high = np.quantile(bootstrap_delta[valid], [0.025, 0.975])
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
                    "n_bootstrap_valid": n_valid,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    return pd.DataFrame(rows)


def _load_strict_label_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _require_mapping(_read_json(path), f"Top level of {path}")
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
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        raise ValueError(f"{path}: strict label artifact is missing {missing_top}")
    expected_top = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
    }
    for key, expected in expected_top.items():
        if payload[key] != expected:
            raise ValueError(f"{path}: {key}={payload[key]!r}, expected {expected!r}")
    model = _require_nonempty_string(payload["model"], f"{path}.model")
    split = _require_nonempty_string(payload["split"], f"{path}.split")
    if split not in {"train", "test"}:
        raise ValueError(f"{path}: split must be train or test, got {split!r}")
    seed = _require_int(payload["seed"], f"{path}.seed")
    tool_scope = _require_nonempty_string(payload["tool_scope"], f"{path}.tool_scope")
    if tool_scope not in {"full", "scoped"}:
        raise ValueError(f"{path}: invalid tool_scope {tool_scope!r}")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{path}: label rows must be a non-empty list")
    if _require_int(payload["n"], f"{path}.n", minimum=1) != len(raw_rows):
        raise ValueError(f"{path}: top-level n differs from len(rows)")

    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
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
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, f"{path} rows[{index}]")
        missing = sorted(required_row - set(row))
        if missing:
            raise ValueError(f"{path} rows[{index}] is missing {missing}")
        task_id = _require_int(row["id"], f"{path} rows[{index}].id")
        if task_id in seen_ids:
            raise ValueError(f"{path}: duplicate label ID {task_id} in split {split}")
        seen_ids.add(task_id)
        for key, expected in {
            "split": split,
            "seed": seed,
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": tool_scope,
        }.items():
            if row[key] != expected:
                raise ValueError(
                    f"{path} rows[{index}].{key}={row[key]!r}, expected {expected!r}"
                )
        difficulty = _require_nonempty_string(
            row["difficulty"], f"{path} rows[{index}].difficulty"
        )
        category = row["category"]
        if category not in TOOL_ACTIONS:
            raise ValueError(
                f"{path} rows[{index}].category must be one of {TOOL_ACTIONS}"
            )
        necessary = row["tool_necessary"]
        if isinstance(necessary, bool) or not isinstance(necessary, int) or necessary not in (0, 1):
            raise TypeError(
                f"{path} rows[{index}].tool_necessary must be integer 0 or 1"
            )
        expected_action = category if necessary else "NONE"
        if row["gold_action"] != expected_action:
            raise ValueError(
                f"{path} rows[{index}].gold_action={row['gold_action']!r}; "
                f"expected {expected_action!r}"
            )
        rows.append(
            {
                "source_file": str(path),
                "id": task_id,
                "split": split,
                "difficulty": difficulty,
                "category": category,
                "tool_necessary": necessary,
            }
        )
    metadata = {
        "schema_version": payload["schema_version"],
        "upstream_commit": payload["upstream_commit"],
        "model": model,
        "split": split,
        "seed": seed,
        "prompt_mode": payload["prompt_mode"],
        "reasoning_mode": payload["reasoning_mode"],
        "tool_scope": tool_scope,
        "protocol_id": payload.get("protocol_id"),
    }
    return metadata, rows


def load_label_distributions(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Combine strict split artifacts and retain split in every output cell."""

    if not paths:
        raise ValueError("At least one strict label artifact is required")
    metadata: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_splits: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        item, item_rows = _load_strict_label_artifact(path)
        if item["split"] in seen_splits:
            raise ValueError(f"Duplicate label artifact for split {item['split']!r}")
        seen_splits.add(item["split"])
        metadata.append(item)
        rows.extend(item_rows)
    invariant_keys = (
        "schema_version",
        "upstream_commit",
        "model",
        "seed",
        "prompt_mode",
        "reasoning_mode",
        "tool_scope",
        "protocol_id",
    )
    reference = metadata[0]
    for item in metadata[1:]:
        for key in invariant_keys:
            if item[key] != reference[key]:
                raise ValueError(
                    f"Label artifacts disagree on {key}: "
                    f"{reference[key]!r} != {item[key]!r}"
                )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["split", "id"]).any():
        raise ValueError("Combined label artifacts contain duplicate (split, id)")
    group_columns = ["split", "difficulty", "category", "tool_necessary"]
    counts = (
        frame.groupby(group_columns, sort=True).size().rename("count").reset_index()
    )
    cell_columns = ["split", "difficulty", "category"]
    totals = counts.groupby(cell_columns, sort=True)["count"].transform("sum")
    counts["cell_total"] = totals
    counts["proportion_within_difficulty_category"] = counts["count"] / totals
    return counts


def load_label_distribution(path: Path | str) -> pd.DataFrame:
    """Backward-compatible strict single-artifact entry point."""

    return load_label_distributions([path])


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
    label_paths: Sequence[Path],
    n_bootstrap: int,
    bootstrap_seed: int,
    expected_seeds: tuple[int, ...] | None,
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
        "labels_file": str(label_paths[0]) if len(label_paths) == 1 else None,
        "labels_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in label_paths
        ],
        "n_rows": int(len(frame)),
        "n_task_ids": int(frame["id"].nunique()),
        "n_settings": int(frame["setting"].nunique()),
        "n_runs": int(len(per_run)),
        "expected_seeds": None if expected_seeds is None else list(expected_seeds),
        "actions": list(ACTIONS),
        "prediction_columns": list(PREDICTIONS),
        "outcome_plot_order": list(OUTCOME_ORDER),
        "error_hierarchy": list(ERROR_HIERARCHY),
        "metric_definitions": METRIC_DEFINITIONS,
        "table_definitions": TABLE_DEFINITIONS,
        "aggregation": (
            "Per-run metrics followed by arithmetic mean and population SD (ddof=0). "
            "Undefined conditional rates are omitted; n_runs is reported per metric."
        ),
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
    labels_paths: Sequence[Path | str] | None = None,
    overwrite: bool = False,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 20260722,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate evaluation outputs and write all statistics and plots."""

    normalized_inputs = [Path(path) for path in output_paths]
    destination = Path(output_dir)
    if labels_path is not None and labels_paths is not None:
        raise ValueError("Pass labels_path or labels_paths, not both")
    normalized_labels = (
        [Path(labels_path)]
        if labels_path is not None
        else [Path(path) for path in labels_paths or ()]
    )
    if destination in normalized_inputs or destination in normalized_labels:
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
    normalized_expected_seeds = (
        None
        if expected_seeds is None
        else validate_expected_seed_panel(frame, expected_seeds)
    )
    per_run = compute_per_run_metrics(frame)
    run_summary = summarize_run_metrics(per_run)
    confusion_counts, confusion_rates = build_confusion_tables(frame)
    recall_summary = build_action_recall_summary(per_run)
    error_counts, error_rates, class_error_rates = build_error_tables(frame)
    needed_category_analysis = build_needed_category_analysis(frame)
    none_analysis = build_none_analysis(frame)
    multicall_summary, multicall_by_gold = build_multicall_tables(frame)
    run_diagnostics, run_diagnostic_summary = build_run_diagnostics(frame)
    gold_final_per_run, gold_final_summary = build_gold_action_final_accuracy(frame)
    tradeoff_per_run, tradeoff_summary = build_current_relative_tradeoff(per_run)
    paired = paired_bootstrap_comparisons(
        frame, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed
    )
    label_distribution = (
        None
        if not normalized_labels
        else load_label_distributions(normalized_labels)
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
        "needed_category_analysis.csv": needed_category_analysis,
        "none_analysis.csv": none_analysis,
        "multicall_summary.csv": multicall_summary,
        "multicall_by_gold.csv": multicall_by_gold,
        "run_diagnostics.csv": run_diagnostics,
        "run_diagnostic_summary.csv": run_diagnostic_summary,
        "gold_action_final_accuracy_per_run.csv": gold_final_per_run,
        "gold_action_final_accuracy_summary.csv": gold_final_summary,
        "current_relative_tradeoff_per_run.csv": tradeoff_per_run,
        "current_relative_tradeoff_summary.csv": tradeoff_summary,
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
        normalized_expected_seeds,
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
