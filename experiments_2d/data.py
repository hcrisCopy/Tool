"""Strict loader and auditor for the official When2Tool Parquet release."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .constants import (
    DIFFICULTIES,
    ENV_TO_CATEGORY,
    EXPECTED_PER_ENV_DIFFICULTY,
    EXPECTED_SPLIT_SIZES,
)


REQUIRED_COLUMNS = (
    "id",
    "difficulty",
    "multi_step",
    "instruction",
    "env_name",
    "tools",
    "parameters",
    "answer",
    "steps",
    "tags",
)


def _decode_json_field(row: dict[str, Any], field: str, expected_type: type) -> Any:
    value = row[field]
    if not isinstance(value, str):
        raise TypeError(f"Row {row['id']} field {field} must be a JSON string")
    decoded = json.loads(value)
    if not isinstance(decoded, expected_type):
        raise TypeError(
            f"Row {row['id']} field {field} must decode to {expected_type.__name__}"
        )
    return decoded


def parquet_path(dataset_root: Path, split: str, multi_hop: bool = False) -> Path:
    config = "multi_hop" if multi_hop else "single_hop"
    matches = sorted((dataset_root / config).glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {config}/{split} Parquet shard, found {matches}"
        )
    return matches[0]


def load_single_hop_split(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    if split not in EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Unsupported split: {split}")
    path = parquet_path(dataset_root, split, multi_hop=False)
    table = pq.read_table(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    unknown = sorted(set(table.column_names) - set(REQUIRED_COLUMNS))
    if unknown:
        raise ValueError(f"{path} has unexpected columns: {unknown}")

    rows = table.to_pylist()
    tasks: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["id"]
        if not isinstance(task_id, int):
            raise TypeError(f"Task id must be int, got {task_id!r}")
        difficulty = row["difficulty"]
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Task {task_id} has invalid difficulty {difficulty!r}")
        if row["multi_step"] is not False:
            raise ValueError(f"Task {task_id} in single_hop must have multi_step=false")
        env_name = row["env_name"]
        if env_name not in ENV_TO_CATEGORY:
            raise ValueError(f"Task {task_id} has unmapped environment {env_name!r}")
        instruction = row["instruction"]
        answer = row["answer"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Task {task_id} has an empty instruction")
        if not isinstance(answer, str):
            raise TypeError(f"Task {task_id} answer must be a string")

        tools = _decode_json_field(row, "tools", list)
        parameters = _decode_json_field(row, "parameters", dict)
        steps = _decode_json_field(row, "steps", list)
        tags = _decode_json_field(row, "tags", list)
        if steps:
            raise ValueError(f"Single-hop task {task_id} unexpectedly contains steps")
        if not tools or not all(isinstance(name, str) and name for name in tools):
            raise ValueError(f"Task {task_id} has invalid tool names: {tools!r}")

        tasks.append(
            {
                "id": task_id,
                "difficulty": difficulty,
                "multi_step": False,
                "instruction": instruction,
                "environments": [
                    {
                        "name": env_name,
                        "tools": tools,
                        "parameters": parameters,
                    }
                ],
                "expected": {"answer": answer},
                "tags": tags,
            }
        )

    validate_single_hop_split(tasks, split)
    return tasks


def validate_single_hop_split(tasks: list[dict[str, Any]], split: str) -> None:
    expected_size = EXPECTED_SPLIT_SIZES[split]
    if len(tasks) != expected_size:
        raise ValueError(f"{split}: expected {expected_size} tasks, got {len(tasks)}")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"{split}: duplicate task ids: {duplicates}")
    instructions = [task["instruction"].strip() for task in tasks]
    if len(instructions) != len(set(instructions)):
        raise ValueError(f"{split}: duplicate instructions found")

    counts = Counter(
        (task["environments"][0]["name"], task["difficulty"])
        for task in tasks
    )
    expected_cell = EXPECTED_PER_ENV_DIFFICULTY[split]
    expected_keys = {
        (env, difficulty)
        for env in ENV_TO_CATEGORY
        for difficulty in DIFFICULTIES
    }
    if set(counts) != expected_keys:
        missing = sorted(expected_keys - set(counts))
        extra = sorted(set(counts) - expected_keys)
        raise ValueError(f"{split}: invalid env/difficulty cells; missing={missing}, extra={extra}")
    wrong = {key: value for key, value in counts.items() if value != expected_cell}
    if wrong:
        raise ValueError(
            f"{split}: expected {expected_cell} rows per env/difficulty cell, got {wrong}"
        )


def base_metadata(tasks: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        {
            "id": task["id"],
            "split": split,
            "env": task["environments"][0]["name"],
            "category": ENV_TO_CATEGORY[task["environments"][0]["name"]],
            "tool_type": ENV_TO_CATEGORY[task["environments"][0]["name"]],
            "difficulty": task["difficulty"],
        }
        for task in tasks
    ]


def dataset_count_rows(tasks_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for split, tasks in tasks_by_split.items():
        for task in tasks:
            env = task["environments"][0]["name"]
            counts[(split, ENV_TO_CATEGORY[env], task["difficulty"])] += 1
    return [
        {"split": split, "category": category, "difficulty": difficulty, "n": count}
        for (split, category, difficulty), count in sorted(counts.items())
    ]


def smoke_subset(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the smallest task id from every env x difficulty cell (45 total)."""

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for task in sorted(tasks, key=lambda item: item["id"]):
        key = (task["environments"][0]["name"], task["difficulty"])
        selected.setdefault(key, task)
    expected = {
        (environment, difficulty)
        for environment in ENV_TO_CATEGORY
        for difficulty in DIFFICULTIES
    }
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(f"Invalid smoke cells; missing={missing}, extra={extra}")
    result = sorted(selected.values(), key=lambda item: item["id"])
    if len(result) != 45:
        raise AssertionError(f"Expected 45 smoke tasks, got {len(result)}")
    return result

