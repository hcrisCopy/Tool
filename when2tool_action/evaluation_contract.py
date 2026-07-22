"""Shared validation for persisted action-evaluation rows.

The evaluator writes several redundant action fields intentionally.  This
module derives them from the routed-event ledger and rejects disagreements so
resume checks and statistics consume the same behavioral contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .constants import ACTIONS


TOOL_ACTIONS = ("A", "B", "C")


@dataclass(frozen=True)
class ActionRowContract:
    gold_action: str
    final_correct: bool
    tool_calls: int
    events: tuple[Mapping[str, Any], ...]
    category_sequence: tuple[str, ...]
    unique_categories: tuple[str, ...]
    pred_action: str
    mixed_category_calls: bool
    invalid_tool_calls: int
    outcome: str


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a JSON boolean, got {value!r}")
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}, got {value}")
    return value


def _require_exact(value: Any, expected: Any, context: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{context}={value!r} disagrees with derived value {expected!r}")


def classify_action_outcome(
    gold_action: str,
    pred_action: str,
    final_correct: bool,
    has_invalid_tool: bool,
) -> str:
    """Apply the preregistered mutually exclusive action-error hierarchy."""

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
    return "success" if final_correct else "correct_category_wrong_answer"


def _check_optional(row: Mapping[str, Any], field: str, expected: Any, context: str) -> None:
    if field in row:
        _require_exact(row[field], expected, f"{context}.{field}")


def validate_action_row(
    row: Mapping[str, Any],
    *,
    context: str,
    require_pred_action: bool,
    strict_event_categories: bool,
    require_formal_diagnostics: bool = False,
) -> ActionRowContract:
    """Validate the statistics contract and all available redundant fields."""

    required = {
        "gold_action",
        "routed_tool_events",
        "tool_calls",
        "final_correct",
    }
    if require_pred_action:
        required.add("pred_action")
    if require_formal_diagnostics:
        required.update({"termination_reason", "tool_parse_failures"})
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"{context} is missing required action fields: {missing}")

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
    events: list[Mapping[str, Any]] = []
    category_sequence: list[str] = []
    for index, event in enumerate(raw_events):
        event_context = f"{context}.routed_tool_events[{index}]"
        if not isinstance(event, Mapping):
            raise TypeError(f"{event_context} must be a JSON object")
        if "category" not in event:
            raise ValueError(f"{event_context} is missing 'category'")
        raw_category = event["category"]
        if strict_event_categories and raw_category not in (*TOOL_ACTIONS, "INVALID"):
            raise ValueError(
                f"{event_context}.category must be one of "
                f"{(*TOOL_ACTIONS, 'INVALID')}, got {raw_category!r}"
            )
        category_sequence.append(
            raw_category if raw_category in TOOL_ACTIONS else "INVALID"
        )
        events.append(event)
    if len(events) != tool_calls:
        raise ValueError(
            f"{context}: len(routed_tool_events)={len(events)} != tool_calls={tool_calls}"
        )

    if require_formal_diagnostics:
        termination_reason = row["termination_reason"]
        if termination_reason not in {"boxed_answer", "max_rounds"}:
            raise ValueError(
                f"{context}.termination_reason must be boxed_answer or max_rounds, "
                f"got {termination_reason!r}"
            )
        _require_int(
            row["tool_parse_failures"],
            f"{context}.tool_parse_failures",
            minimum=0,
        )
        for index, event in enumerate(events):
            if "result" not in event:
                raise ValueError(
                    f"{context}.routed_tool_events[{index}] is missing 'result'"
                )
        if "episode_done" in row:
            episode_done = _require_bool(row["episode_done"], f"{context}.episode_done")
            expected_termination = "boxed_answer" if episode_done else "max_rounds"
            _require_exact(
                termination_reason,
                expected_termination,
                f"{context}.termination_reason",
            )

    pred_action = "NONE" if not category_sequence else category_sequence[0]
    unique_categories = list(dict.fromkeys(category_sequence))
    mixed_category_calls = len(unique_categories) > 1
    invalid_tool_calls = sum(category == "INVALID" for category in category_sequence)
    outcome = classify_action_outcome(
        gold_action,
        pred_action,
        final_correct,
        invalid_tool_calls > 0,
    )

    _check_optional(row, "pred_action", pred_action, context)
    _check_optional(row, "total_tool_calls", tool_calls, context)
    _check_optional(row, "tool_call_categories", category_sequence, context)
    _check_optional(row, "unique_tool_call_categories", unique_categories, context)
    _check_optional(row, "n_tool_call_categories", len(unique_categories), context)
    _check_optional(row, "mixed_category_calls", mixed_category_calls, context)
    _check_optional(row, "invalid_tool_calls", invalid_tool_calls, context)
    _check_optional(row, "first_tool_category", pred_action if events else None, context)
    _check_optional(row, "error_type", outcome, context)

    first = events[0] if events else None
    if "first_tool_name" in row:
        _check_optional(
            row,
            "first_tool_name",
            first.get("tool_name") if first is not None else None,
            context,
        )
    if "first_tool_environment" in row:
        _check_optional(
            row,
            "first_tool_environment",
            first.get("environment") if first is not None else None,
            context,
        )
    if "first_arguments_valid" in row:
        expected_arguments_valid = bool(
            first is not None and first.get("arguments_valid") is True
        )
        _check_optional(
            row,
            "first_arguments_valid",
            expected_arguments_valid,
            context,
        )
    if "first_env_correct" in row and "gold_env_name" in row:
        _check_optional(
            row,
            "first_env_correct",
            bool(first is not None and first.get("environment") == row["gold_env_name"]),
            context,
        )
    if "exact_tool_allowed" in row and "gold_tools" in row:
        gold_tools = row["gold_tools"]
        if not isinstance(gold_tools, list):
            raise TypeError(f"{context}.gold_tools must be a list")
        _check_optional(
            row,
            "exact_tool_allowed",
            bool(first is not None and first.get("tool_name") in set(gold_tools)),
            context,
        )

    return ActionRowContract(
        gold_action=gold_action,
        final_correct=final_correct,
        tool_calls=tool_calls,
        events=tuple(events),
        category_sequence=tuple(category_sequence),
        unique_categories=tuple(unique_categories),
        pred_action=pred_action,
        mixed_category_calls=mixed_category_calls,
        invalid_tool_calls=invalid_tool_calls,
        outcome=outcome,
    )
