"""Fail-closed construction of full-menu action-SFT trajectories.

The successful tool demonstrations are generated with the scoped menu because
that is the original When2Tool setting.  They must *not* be trained with that
prompt: this module verifies the saved scoped prompt exactly, removes it, and
grafts only the assistant/tool-response suffix onto a freshly reconstructed
canonical full-menu ``current`` prompt.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .constants import (
    ACTIONS,
    DIFFICULTIES,
    ENV_TO_CATEGORY,
    EXPECTED_SPLIT_SIZES,
    SCHEMA_VERSION,
    UPSTREAM_COMMIT,
)
from .io_utils import atomic_write_json, canonical_json_sha256, sha256_bytes
from .runtime import EvaluationSetting, initial_messages_and_tools


SFT_SCHEMA_VERSION = "when2tool-action-sft-v1"
SFT_MANIFEST_SCHEMA_VERSION = "when2tool-action-sft-manifest-v1"
EXPECTED_SFT_SEED = 0
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class SFTBuildResult:
    records: list[dict[str, Any]]
    manifest: dict[str, Any]


InitialBuilder = Callable[..., tuple[list[dict[str, str]], Any]]


def read_json_artifact(path: Path | str) -> Any:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact: {source}") from error


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _require_rows(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise TypeError(f"{context}[{index}] must be an object")
        rows.append(row)
    return rows


def _index_rows(
    rows: Sequence[dict[str, Any]],
    *,
    context: str,
    expected_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise TypeError(f"{context} contains a non-integer task id {task_id!r}")
        if task_id in indexed:
            raise ValueError(f"{context} contains duplicate task id {task_id}")
        indexed[task_id] = row
    expected = set(expected_ids)
    if set(indexed) != expected:
        raise ValueError(
            f"{context} task IDs differ: "
            f"missing={sorted(expected - set(indexed))[:20]}, "
            f"extra={sorted(set(indexed) - expected)[:20]}"
        )
    return indexed


def _validate_labels_artifact(
    artifact: Any, expected_ids: Sequence[int], *, expected_model_slug: str
) -> dict[int, dict[str, Any]]:
    root = _require_object(artifact, "labels artifact")
    expected_root = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": expected_model_slug,
        "split": "train",
        "seed": EXPECTED_SFT_SEED,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
    }
    for key, expected in expected_root.items():
        if root.get(key) != expected:
            raise ValueError(f"labels artifact {key} must be {expected!r}")
    if root.get("n") != len(expected_ids):
        raise ValueError("labels artifact n differs from the train task count")
    rows = _index_rows(
        _require_rows(root.get("rows"), "labels.rows"),
        context="labels.rows",
        expected_ids=expected_ids,
    )
    present: set[str] = set()
    for task_id, row in rows.items():
        action = row.get("gold_action")
        if action not in ACTIONS:
            raise ValueError(f"Label task {task_id} has invalid gold_action {action!r}")
        present.add(action)
        necessary = row.get("tool_necessary")
        if necessary not in {0, 1} or isinstance(necessary, bool):
            raise ValueError(f"Label task {task_id} has invalid tool_necessary")
        expected_action = row.get("category") if necessary == 1 else "NONE"
        if action != expected_action:
            raise ValueError(f"Label task {task_id} has inconsistent gold_action")
        row_contract = {
            "split": "train",
            "seed": EXPECTED_SFT_SEED,
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": "full",
        }
        for key, expected in row_contract.items():
            if row.get(key) != expected:
                raise ValueError(f"Label task {task_id} {key} must be {expected!r}")
        if row.get("no_tool_correct") not in {0, 1} or isinstance(
            row.get("no_tool_correct"), bool
        ):
            raise ValueError(f"Label task {task_id} has invalid no_tool_correct")
        if row["no_tool_correct"] != 1 - necessary:
            raise ValueError(f"Label task {task_id} no_tool_correct is inconsistent")
    if present != set(ACTIONS):
        raise ValueError(
            f"Training labels must contain every action; missing={sorted(set(ACTIONS) - present)}"
        )
    return rows


def _validate_no_tool_artifact(
    artifact: Any, expected_ids: Sequence[int]
) -> dict[int, dict[str, Any]]:
    root = _require_object(artifact, "full hard-no-tool artifact")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Full hard-no-tool artifact schema_version differs")
    if root.get("split") != "train" or root.get("seed") != EXPECTED_SFT_SEED:
        raise ValueError("Full hard-no-tool artifact must be train split, seed 0")
    setting = root.get("setting")
    if not isinstance(setting, str) or "hard_no_tool" not in setting:
        raise ValueError("Full hard-no-tool artifact has the wrong setting")
    if "fulltools" not in setting:
        raise ValueError("Hard-no-tool trajectories must use the full menu")
    rows = _index_rows(
        _require_rows(root.get("rows"), "hard_no_tool.rows"),
        context="hard_no_tool.rows",
        expected_ids=expected_ids,
    )
    for task_id, row in rows.items():
        checks = {
            "schema_version": SCHEMA_VERSION,
            "seed": EXPECTED_SFT_SEED,
            "tool_scope": "full",
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Hard-no-tool task {task_id} {key} must be {expected!r}"
                )
    return rows


def _validate_scoped_artifact(
    artifact: Any, expected_ids: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], str]:
    root = _require_object(artifact, "scoped trajectory artifact")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Scoped trajectory schema_version differs")
    if root.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Scoped trajectory upstream commit differs")
    config = _require_object(root.get("config"), "scoped.config")
    expected_config = {
        "tool_scope": "scoped",
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "record_mode": "full",
        "seeds": [EXPECTED_SFT_SEED],
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"scoped.config.{key} must be {expected!r}")
    runs = _require_rows(root.get("runs"), "scoped.runs")
    if len(runs) != 1:
        raise ValueError("Scoped SFT source must contain exactly the seed-0 run")
    run = runs[0]
    if run.get("seed") != EXPECTED_SFT_SEED:
        raise ValueError("Scoped SFT source run must use seed 0")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TypeError("Scoped SFT source run_id must be non-empty")
    rows = _index_rows(
        _require_rows(run.get("rows"), "scoped.runs[0].rows"),
        context="scoped.runs[0].rows",
        expected_ids=expected_ids,
    )
    for task_id, row in rows.items():
        checks = {
            "schema_version": SCHEMA_VERSION,
            "seed": EXPECTED_SFT_SEED,
            "tool_scope": "scoped",
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(f"Scoped task {task_id} {key} must be {expected!r}")
    return rows, run_id


def _validate_task_label(task: dict[str, Any], label: dict[str, Any]) -> None:
    task_id = task["id"]
    expected_pairs = (
        ("difficulty", task.get("difficulty")),
        ("category", task.get("category")),
        ("gold_env_name", task.get("gold_env_name")),
    )
    for key, expected in expected_pairs:
        if label.get(key) != expected:
            raise ValueError(f"Task {task_id} label {key} differs from task data")
    category = task.get("category")
    env = task.get("gold_env_name")
    if env not in ENV_TO_CATEGORY or category != ENV_TO_CATEGORY[env]:
        raise ValueError(f"Task {task_id} has invalid environment/category metadata")
    if task.get("difficulty") not in DIFFICULTIES:
        raise ValueError(f"Task {task_id} has invalid difficulty")


def _validate_source_metadata(
    task: dict[str, Any], row: dict[str, Any], *, context: str
) -> None:
    task_id = task["id"]
    for key in ("difficulty", "category", "gold_env_name"):
        if row.get(key) != task.get(key):
            raise ValueError(f"{context} task {task_id} {key} differs from train data")


def _boxed_success(row: Mapping[str, Any]) -> bool:
    return (
        row.get("episode_done") is True
        and row.get("termination_reason") == "boxed_answer"
        and "\\boxed{" in str(row.get("final_response", ""))
    )


def _tool_tail(
    output: Any,
    scoped_initial: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Return a validated assistant/tool-response suffix and a drop reason."""

    if not isinstance(output, list) or len(output) <= len(scoped_initial):
        return None, "dropped_tool_malformed_output"
    prefix = output[: len(scoped_initial)]
    if prefix != list(scoped_initial):
        return None, "dropped_tool_scoped_prefix_mismatch"
    tail: list[dict[str, str]] = []
    assistant_count = 0
    tool_response_count = 0
    for message in output[len(scoped_initial) :]:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            return None, "dropped_tool_malformed_tail"
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            return None, "dropped_tool_malformed_tail"
        if role == "assistant":
            assistant_count += 1
        elif role == "tool":
            tool_response_count += 1
        elif (
            role == "user"
            and content.startswith("<tool_response>\n")
            and content.endswith("\n</tool_response>")
        ):
            tool_response_count += 1
        else:
            # In particular, never train evaluator retry/continue messages.
            return None, "dropped_tool_nontrajectory_feedback"
        tail.append({"role": role, "content": content})
    if assistant_count < 2 or tool_response_count < 1:
        return None, "dropped_tool_malformed_tail"
    if tail[0]["role"] != "assistant" or tail[-1]["role"] != "assistant":
        return None, "dropped_tool_malformed_tail"
    return tail, None


def _row_has_invalid_call(row: Mapping[str, Any]) -> bool:
    if row.get("invalid_tool_calls") != 0:
        return True
    categories = row.get("tool_call_categories")
    if isinstance(categories, list) and "INVALID" in categories:
        return True
    events = row.get("routed_tool_events")
    if isinstance(events, list):
        return any(
            isinstance(event, dict) and event.get("category") == "INVALID"
            for event in events
        )
    return False


def _group_decisions(
    decisions: Sequence[dict[str, Any]], key: str, expected_values: Sequence[str]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in expected_values:
        selected = [row for row in decisions if row[key] == value]
        reasons = Counter(row["reason"] for row in selected)
        output[value] = {
            "total": len(selected),
            "retained": sum(row["decision"] == "retained" for row in selected),
            "dropped": sum(row["decision"] == "dropped" for row in selected),
            "reasons": dict(sorted(reasons.items())),
        }
    return output


def _decision_manifest(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts = Counter(row["reason"] for row in decisions)
    retained = sum(row["decision"] == "retained" for row in decisions)
    dropped = len(decisions) - retained
    return {
        "total": len(decisions),
        "retained": retained,
        "dropped": dropped,
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_action": _group_decisions(decisions, "gold_action", ACTIONS),
        "by_environment": _group_decisions(
            decisions, "gold_env_name", tuple(sorted(ENV_TO_CATEGORY))
        ),
        "by_difficulty": _group_decisions(decisions, "difficulty", DIFFICULTIES),
        "task_decisions": decisions,
    }


def _source_sha(sources: Mapping[str, Mapping[str, Any]], name: str) -> str:
    entry = sources.get(name)
    if not isinstance(entry, Mapping):
        raise KeyError(f"Missing source provenance for {name}")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"Source provenance {name}.sha256 is invalid")
    return digest


def build_sft_trajectories(
    tasks: Sequence[dict[str, Any]],
    scoped_tasks: Sequence[dict[str, Any]],
    labels_artifact: Any,
    no_tool_artifact: Any,
    scoped_artifact: Any,
    *,
    system_prompt: str,
    source_provenance: Mapping[str, Mapping[str, Any]],
    expected_model_slug: str,
    expected_full_menu_sha256: str,
    runtime_provenance: Mapping[str, Any],
    expected_task_count: int = EXPECTED_SPLIT_SIZES["train"],
    initial_builder: InitialBuilder = initial_messages_and_tools,
) -> SFTBuildResult:
    """Build filtered trajectories without ever copying a scoped prompt."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be non-empty")
    if not isinstance(expected_model_slug, str) or not expected_model_slug.strip():
        raise ValueError("expected_model_slug must be non-empty")
    if expected_task_count <= 0 or len(tasks) != expected_task_count:
        raise ValueError(
            f"SFT construction requires {expected_task_count} train tasks, got {len(tasks)}"
        )
    if _SHA256_RE.fullmatch(expected_full_menu_sha256) is None:
        raise ValueError("expected_full_menu_sha256 must be a lowercase SHA-256")
    expected_sources = {
        "data",
        "scoped_data",
        "labels",
        "no_tool_generations",
        "scoped_trajectories",
    }
    if set(source_provenance) != expected_sources:
        raise ValueError(
            "SFT source provenance keys differ: "
            f"missing={sorted(expected_sources - set(source_provenance))}, "
            f"extra={sorted(set(source_provenance) - expected_sources)}"
        )
    if not isinstance(runtime_provenance, Mapping):
        raise TypeError("runtime_provenance must be an object")
    runtime_sha256 = runtime_provenance.get("sha256")
    runtime_commit = runtime_provenance.get("project_git_commit")
    runtime_file = runtime_provenance.get("file")
    if (
        not isinstance(runtime_sha256, str)
        or _SHA256_RE.fullmatch(runtime_sha256) is None
    ):
        raise ValueError("runtime_provenance.sha256 is invalid")
    if (
        not isinstance(runtime_commit, str)
        or _COMMIT_RE.fullmatch(runtime_commit) is None
    ):
        raise ValueError("runtime_provenance.project_git_commit is invalid")
    if not isinstance(runtime_file, str) or not runtime_file:
        raise ValueError("runtime_provenance.file must be non-empty")
    task_ids = [task.get("id") for task in tasks]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in task_ids):
        raise TypeError("Every SFT task id must be an integer")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate SFT train task IDs")
    if any(task.get("tool_scope") != "full" for task in tasks):
        raise ValueError("SFT task data must be the full-menu train split")
    if len(scoped_tasks) != expected_task_count or any(
        task.get("tool_scope") != "scoped" for task in scoped_tasks
    ):
        raise ValueError(
            "SFT scoped source data must be the complete scoped train split"
        )
    scoped_task_ids = [task.get("id") for task in scoped_tasks]
    if scoped_task_ids != task_ids:
        raise ValueError("Scoped/full train task IDs or order differ")
    for full_task, scoped_task in zip(tasks, scoped_tasks, strict=True):
        normalized_full = deepcopy(full_task)
        normalized_scoped = deepcopy(scoped_task)
        normalized_full.pop("tool_scope", None)
        normalized_scoped.pop("tool_scope", None)
        if normalized_full != normalized_scoped:
            raise ValueError(
                f"Scoped/full train task {full_task['id']} differs beyond tool_scope"
            )

    labels = _validate_labels_artifact(
        labels_artifact, task_ids, expected_model_slug=expected_model_slug
    )
    no_tool_rows = _validate_no_tool_artifact(no_tool_artifact, task_ids)
    scoped_rows, scoped_run_id = _validate_scoped_artifact(scoped_artifact, task_ids)
    source_hashes = {
        name: _source_sha(source_provenance, name)
        for name in (
            "data",
            "scoped_data",
            "labels",
            "no_tool_generations",
            "scoped_trajectories",
        )
    }
    scoped_config = scoped_artifact["config"]
    scoped_bindings = {
        "model": expected_model_slug,
        "data_sha256": source_hashes["scoped_data"],
        "labels_sha256": source_hashes["labels"],
        "runtime_provenance_sha256": runtime_sha256,
        "project_git_commit": runtime_commit,
        "full_menu_sha256": expected_full_menu_sha256,
        "task_ids_sha256": canonical_json_sha256(task_ids),
        "smoke": False,
    }
    for key, expected in scoped_bindings.items():
        if scoped_config.get(key) != expected:
            raise ValueError(f"scoped.config.{key} must be {expected!r}")
    input_bundle_sha256 = canonical_json_sha256(source_hashes)

    full_setting = EvaluationSetting(
        name="sft_full_current_no_reasoning",
        tool_scope="full",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="off",
    )
    scoped_setting = EvaluationSetting(
        name="sft_scoped_prefix_validation",
        tool_scope="scoped",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="off",
    )
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    observed_menu_hashes: set[str] = set()

    for task, scoped_task in zip(tasks, scoped_tasks, strict=True):
        task_id = task["id"]
        label = labels[task_id]
        no_tool_row = no_tool_rows[task_id]
        scoped_row = scoped_rows[task_id]
        _validate_task_label(task, label)
        _validate_source_metadata(task, no_tool_row, context="hard-no-tool")
        _validate_source_metadata(task, scoped_row, context="scoped")
        action = label["gold_action"]
        no_tool_correct = bool(no_tool_row.get("final_correct"))
        if no_tool_correct != (action == "NONE"):
            raise ValueError(
                f"Task {task_id} hard-no-tool result disagrees with frozen gold action"
            )
        if label.get("final_response") != no_tool_row.get("final_response"):
            raise ValueError(
                f"Task {task_id} label response differs from hard-no-tool source"
            )
        if scoped_row.get("gold_action") != action:
            raise ValueError(f"Task {task_id} scoped source uses different labels")
        decision = {
            "id": task_id,
            "gold_action": action,
            "gold_env_name": task["gold_env_name"],
            "difficulty": task["difficulty"],
            "decision": "dropped",
            "reason": "",
        }

        full_initial, full_built = initial_builder(
            task, system_prompt=system_prompt, setting=full_setting
        )
        menu_hash = getattr(full_built, "menu_sha256", None)
        schemas = getattr(full_built, "schemas", None)
        if menu_hash != expected_full_menu_sha256:
            raise ValueError(f"Task {task_id} reconstructed a non-canonical full menu")
        if not isinstance(schemas, list) or len(schemas) != 33:
            raise ValueError(f"Task {task_id} full menu does not contain 33 tools")
        observed_menu_hashes.add(menu_hash)
        source_kind: str
        source_row: dict[str, Any]
        source_hash: str
        source_run: str

        if action == "NONE":
            source_kind = "full_hard_no_tool_direct_answer"
            source_row = no_tool_row
            source_hash = source_hashes["no_tool_generations"]
            source_run = str(no_tool_row.get("run_id", "labels_seed_0"))
            if no_tool_row.get("final_correct") is not True:
                decision["reason"] = "dropped_none_direct_answer_incorrect"
            elif no_tool_row.get("total_tool_calls") != 0 or no_tool_row.get(
                "routed_tool_events"
            ):
                decision["reason"] = "dropped_none_used_tool"
            elif _row_has_invalid_call(no_tool_row):
                decision["reason"] = "dropped_none_invalid_call"
            elif not _boxed_success(no_tool_row):
                decision["reason"] = "dropped_none_not_boxed"
            else:
                final_response = no_tool_row.get("final_response")
                if not isinstance(final_response, str) or not final_response:
                    decision["reason"] = "dropped_none_missing_response"
                else:
                    messages = [
                        *deepcopy(full_initial),
                        {"role": "assistant", "content": final_response},
                    ]
                    decision.update(
                        decision="retained", reason="retained_none_direct_answer"
                    )
        else:
            source_kind = "scoped_current_success_tail"
            source_row = scoped_row
            source_hash = source_hashes["scoped_trajectories"]
            source_run = scoped_run_id
            if scoped_row.get("final_correct") is not True:
                decision["reason"] = "dropped_tool_final_incorrect"
            elif (
                not isinstance(scoped_row.get("total_tool_calls"), int)
                or scoped_row["total_tool_calls"] < 1
            ):
                decision["reason"] = "dropped_tool_no_routed_call"
            elif scoped_row.get("first_tool_category") != action:
                decision["reason"] = "dropped_tool_first_category_mismatch"
            elif _row_has_invalid_call(scoped_row):
                decision["reason"] = "dropped_tool_invalid_call"
            elif not _boxed_success(scoped_row):
                decision["reason"] = "dropped_tool_not_boxed"
            else:
                scoped_initial, _scoped_built = initial_builder(
                    scoped_task, system_prompt=system_prompt, setting=scoped_setting
                )
                tail, drop_reason = _tool_tail(scoped_row.get("output"), scoped_initial)
                if drop_reason is not None:
                    decision["reason"] = drop_reason
                else:
                    assert tail is not None
                    messages = [*deepcopy(full_initial), *tail]
                    decision.update(
                        decision="retained", reason="retained_tool_trajectory"
                    )

        decisions.append(decision)
        if decision["decision"] != "retained":
            continue
        message_digest = canonical_json_sha256(messages)
        input_digest = canonical_json_sha256(
            {
                "full_task": task,
                "scoped_task": scoped_task,
                "label": label,
                "source_row": source_row,
            }
        )
        records.append(
            {
                "schema_version": SFT_SCHEMA_VERSION,
                "id": task_id,
                "gold_action": action,
                "category": task["category"],
                "difficulty": task["difficulty"],
                "gold_env_name": task["gold_env_name"],
                "messages": messages,
                "messages_sha256": message_digest,
                "full_menu_sha256": menu_hash,
                "input_sha256": input_digest,
                "provenance": {
                    "source_kind": source_kind,
                    "source_artifact_sha256": source_hash,
                    "source_run_id": source_run,
                    "source_seed": EXPECTED_SFT_SEED,
                    "input_bundle_sha256": input_bundle_sha256,
                    "runtime_provenance_sha256": runtime_sha256,
                    "full_prompt_messages_sha256": canonical_json_sha256(full_initial),
                },
            }
        )

    if observed_menu_hashes != {expected_full_menu_sha256}:
        raise AssertionError("Full-menu hash changed across SFT tasks")
    retained_actions = {record["gold_action"] for record in records}
    if retained_actions != set(ACTIONS):
        raise ValueError(
            "Filtered SFT data must retain every action; "
            f"missing={sorted(set(ACTIONS) - retained_actions)}"
        )
    decision_summary = _decision_manifest(decisions)
    if decision_summary["retained"] != len(records):
        raise AssertionError("SFT decision counts do not match retained records")
    manifest: dict[str, Any] = {
        "schema_version": SFT_MANIFEST_SCHEMA_VERSION,
        "protocol": {
            "split": "train",
            "source_task_count": expected_task_count,
            "source_seed": EXPECTED_SFT_SEED,
            "target_tool_scope": "full",
            "target_prompt_mode": "current",
            "target_reasoning_mode": "no_reasoning",
            "tool_source_scope": "scoped",
            "none_source_prompt_mode": "hard_no_tool",
            "filtering": (
                "NONE requires a correct boxed no-call answer; A/B/C require a "
                "correct boxed scoped trajectory whose first category is gold and "
                "which contains no INVALID call"
            ),
        },
        "full_menu_sha256": expected_full_menu_sha256,
        "task_ids_sha256": canonical_json_sha256(task_ids),
        "input_bundle_sha256": input_bundle_sha256,
        "inputs": {name: dict(value) for name, value in source_provenance.items()},
        "runtime_provenance": dict(runtime_provenance),
        "decisions": decision_summary,
    }
    return SFTBuildResult(records=records, manifest=manifest)


def _jsonl_payload(records: Sequence[dict[str, Any]]) -> bytes:
    if not records:
        raise ValueError("Cannot write an empty SFT JSONL")
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_sft_artifacts(
    output_path: Path | str,
    manifest_path: Path | str,
    result: SFTBuildResult,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically publish the JSONL and its auditable filtering manifest."""

    output = Path(output_path).resolve()
    manifest_target = Path(manifest_path).resolve()
    if output == manifest_target:
        raise ValueError("SFT JSONL and manifest paths must differ")
    for path in (output, manifest_target):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        if path.exists() and not path.is_file():
            raise IsADirectoryError(path)
    payload = _jsonl_payload(result.records)
    jsonl_sha256 = sha256_bytes(payload)
    manifest = deepcopy(result.manifest)
    manifest["output"] = {
        "file": output.name,
        "n_records": len(result.records),
        "bytes": len(payload),
        "sha256": jsonl_sha256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        atomic_write_json(manifest_target, manifest, overwrite=overwrite)
    except Exception:
        # Do not leave an apparently usable JSONL without its contract receipt.
        if output.is_file() and sha256_bytes(output.read_bytes()) == jsonl_sha256:
            output.unlink()
        raise
    return manifest
