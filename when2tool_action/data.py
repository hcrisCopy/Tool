"""Strict loader and category/full-menu materializer for official Parquet data."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .constants import (
    CATEGORY_NAMES,
    DIFFICULTIES,
    ENV_TO_CATEGORY,
    EXPECTED_PER_ENV_DIFFICULTY,
    EXPECTED_SPLIT_SIZES,
    SCHEMA_VERSION,
    TAXONOMY_NOTE,
)
from .io_utils import atomic_write_json, canonical_json_sha256
from .upstream import build_environments, full_menu_sha256


REQUIRED_COLUMNS = {
    "id", "difficulty", "multi_step", "instruction", "env_name", "tools",
    "parameters", "answer", "steps", "tags",
}


def _decode(row: dict[str, Any], field: str, kind: type) -> Any:
    raw = row[field]
    if not isinstance(raw, str):
        raise TypeError(f"Task {row.get('id')} {field} is not encoded JSON")
    value = json.loads(raw)
    if not isinstance(value, kind):
        raise TypeError(f"Task {row.get('id')} {field} is not {kind.__name__}")
    return value


def parquet_path(dataset_root: Path, split: str) -> Path:
    matches = sorted((dataset_root / "single_hop").glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one single_hop/{split} Parquet shard, found {matches}"
        )
    return matches[0]


def load_split(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    if split not in EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Unsupported split {split}")
    path = parquet_path(dataset_root, split)
    table = pq.read_table(path)
    if set(table.column_names) != REQUIRED_COLUMNS:
        raise ValueError(
            f"Unexpected Parquet columns; missing={sorted(REQUIRED_COLUMNS-set(table.column_names))}, "
            f"extra={sorted(set(table.column_names)-REQUIRED_COLUMNS)}"
        )
    tasks: list[dict[str, Any]] = []
    for row in table.to_pylist():
        task_id = row["id"]
        if not isinstance(task_id, int):
            raise TypeError(f"Invalid task id {task_id!r}")
        env_name = row["env_name"]
        if env_name not in ENV_TO_CATEGORY:
            raise ValueError(f"Task {task_id} unknown env {env_name}")
        if row["difficulty"] not in DIFFICULTIES or row["multi_step"] is not False:
            raise ValueError(f"Task {task_id} violates single-hop split contract")
        tools = _decode(row, "tools", list)
        parameters = _decode(row, "parameters", dict)
        steps = _decode(row, "steps", list)
        tags = _decode(row, "tags", list)
        if steps or not tools or not all(isinstance(name, str) and name for name in tools):
            raise ValueError(f"Task {task_id} has invalid single-hop tools/steps")
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise ValueError(f"Task {task_id} has empty instruction")
        if not isinstance(row["answer"], str):
            raise TypeError(f"Task {task_id} answer is not a string")
        category = ENV_TO_CATEGORY[env_name]
        tasks.append(
            {
                "id": task_id,
                "difficulty": row["difficulty"],
                "multi_step": False,
                "instruction": row["instruction"],
                "environments": [
                    {"name": env_name, "tools": tools, "parameters": parameters}
                ],
                "expected": {"answer": row["answer"]},
                "tags": tags,
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "gold_env_name": env_name,
                "gold_tools": list(tools),
            }
        )
    validate_split(tasks, split)
    return tasks


def validate_split(tasks: list[dict[str, Any]], split: str) -> None:
    if len(tasks) != EXPECTED_SPLIT_SIZES[split]:
        raise ValueError(f"{split}: got {len(tasks)} tasks")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split}: duplicate IDs")
    instructions = [task["instruction"].strip() for task in tasks]
    if len(instructions) != len(set(instructions)):
        raise ValueError(f"{split}: duplicate instructions")
    counts = Counter(
        (task["gold_env_name"], task["difficulty"]) for task in tasks
    )
    expected_keys = {
        (env, difficulty) for env in ENV_TO_CATEGORY for difficulty in DIFFICULTIES
    }
    if set(counts) != expected_keys:
        raise ValueError(f"{split}: missing or extra env/difficulty cells")
    wrong = {
        key: value
        for key, value in counts.items()
        if value != EXPECTED_PER_ENV_DIFFICULTY[split]
    }
    if wrong:
        raise ValueError(f"{split}: wrong cell sizes {wrong}")
    for task in tasks:
        if task["category"] != ENV_TO_CATEGORY[task["gold_env_name"]]:
            raise ValueError(f"Task {task['id']} category mismatch")


def write_augmented_data(
    dataset_root: Path, output_dir: Path, *, overwrite: bool = False
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_note": TAXONOMY_NOTE,
        "full_menu_sha256": full_menu_sha256(),
        "splits": {},
    }
    for split in ("train", "test"):
        tasks = load_split(dataset_root, split)
        # Validate both environment construction paths on every task now, rather
        # than discovering malformed parameters during generation.
        for task in tasks:
            build_environments(task, "scoped")
            built = build_environments(task, "full")
            if built.menu_sha256 != manifest["full_menu_sha256"]:
                raise AssertionError("Full menu hash varies by task")
        scoped_rows = [dict(deepcopy(task), tool_scope="scoped") for task in tasks]
        full_rows = [dict(deepcopy(task), tool_scope="full") for task in tasks]
        category_path = output_dir / f"tasks_v1_{split}_category.json"
        full_path = output_dir / f"tasks_v1_{split}_fulltools_category.json"
        atomic_write_json(category_path, scoped_rows, overwrite=overwrite)
        atomic_write_json(full_path, full_rows, overwrite=overwrite)
        manifest["splits"][split] = {
            "n": len(tasks),
            "ids_sha256": canonical_json_sha256([task["id"] for task in tasks]),
            "category_file": category_path.name,
            "fulltools_file": full_path.name,
        }
    atomic_write_json(output_dir / "data_manifest.json", manifest, overwrite=overwrite)
    return manifest


def load_task_json(path: Path, *, expected_scope: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise TypeError(f"{path} must contain a non-empty task list")
    ids: set[int] = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), int):
            raise TypeError(f"Malformed task in {path}")
        if task["id"] in ids:
            raise ValueError(f"Duplicate task ID {task['id']} in {path}")
        ids.add(task["id"])
        env = task.get("gold_env_name")
        if env not in ENV_TO_CATEGORY or task.get("category") != ENV_TO_CATEGORY[env]:
            raise ValueError(f"Task {task['id']} has invalid category metadata")
        if expected_scope is not None and task.get("tool_scope") != expected_scope:
            raise ValueError(f"Task {task['id']} scope mismatch")
    return tasks


def smoke_subset(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for task in sorted(tasks, key=lambda row: row["id"]):
        selected.setdefault((task["gold_env_name"], task["difficulty"]), task)
    if len(selected) != 45:
        raise ValueError(f"Smoke subset must cover 45 cells, got {len(selected)}")
    return sorted(selected.values(), key=lambda row: row["id"])
