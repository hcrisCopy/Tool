"""Model-specific hard-no-tool label construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .constants import ACTIONS, SCHEMA_VERSION, UPSTREAM_COMMIT


def build_label_rows(
    evaluation_rows: list[dict[str, Any]], *, split: str, seed: int, tool_scope: str
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in evaluation_rows:
        task_id = row.get("id")
        if not isinstance(task_id, int) or task_id in seen:
            raise ValueError(f"Invalid or duplicate evaluated task ID {task_id}")
        seen.add(task_id)
        if row.get("total_tool_calls") != 0 or row.get("routed_tool_events"):
            raise AssertionError(f"Hard-no-tool task {task_id} executed a tool")
        category = row.get("category")
        if category not in {"A", "B", "C"}:
            raise ValueError(f"Task {task_id} has invalid category {category}")
        no_tool_correct = int(bool(row.get("final_correct")))
        tool_necessary = 1 - no_tool_correct
        gold_action = category if tool_necessary else "NONE"
        if gold_action not in ACTIONS:
            raise AssertionError("Constructed action is outside the frozen label set")
        labels.append(
            {
                "id": task_id,
                "split": split,
                "difficulty": row["difficulty"],
                "category": category,
                "category_name": row["category_name"],
                "gold_env_name": row["gold_env_name"],
                "gold_tools": deepcopy(row["gold_tools"]),
                "no_tool_correct": no_tool_correct,
                "tool_necessary": tool_necessary,
                "gold_action": gold_action,
                "seed": seed,
                "prompt_mode": "hard_no_tool",
                "reasoning_mode": "no_reasoning",
                "tool_scope": tool_scope,
                "final_response": row.get("final_response", ""),
            }
        )
    return labels


def label_artifact(
    rows: list[dict[str, Any]], *, model_slug: str, split: str, seed: int, tool_scope: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": model_slug,
        "split": split,
        "seed": seed,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": tool_scope,
        "n": len(rows),
        "rows": rows,
    }
