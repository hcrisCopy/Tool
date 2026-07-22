from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.io_utils import canonical_json_sha256
from when2tool_action.masked_lora import load_sft_records
from when2tool_action.sft import build_sft_trajectories, write_sft_artifacts


MENU_SHA = "f" * 64
SOURCE_PROVENANCE = {
    "data": {"file": "train.json", "sha256": "1" * 64},
    "scoped_data": {"file": "train_scoped.json", "sha256": "7" * 64},
    "labels": {"file": "labels.json", "sha256": "2" * 64},
    "no_tool_generations": {"file": "no_tool.json", "sha256": "3" * 64},
    "scoped_trajectories": {"file": "scoped.json", "sha256": "4" * 64},
}
RUNTIME_PROVENANCE = {
    "file": "runtime_provenance.json",
    "sha256": "5" * 64,
    "project_git_commit": "6" * 40,
}


def _task(task_id: int, action: str) -> dict:
    category, env, tools = {
        "NONE": ("A", "CalculatorEnv", ["evaluate_expression"]),
        "A": ("A", "CalculatorEnv", ["evaluate_expression"]),
        "B": ("B", "RetrieverEnv", ["search_corpus"]),
        "C": ("C", "CodeExecutorEnv", ["run_code"]),
    }[action]
    return {
        "id": task_id,
        "difficulty": ("easy", "medium", "hard")[task_id % 3],
        "instruction": f"task {task_id}",
        "environments": [{"name": env, "tools": tools, "parameters": {}}],
        "expected": {"answer": str(task_id)},
        "category": category,
        "category_name": "test",
        "gold_env_name": env,
        "gold_tools": tools,
        "tool_scope": "full",
    }


def _initial_builder(task: dict, *, system_prompt: str, setting: object):
    scope = setting.tool_scope
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{scope.upper()} CURRENT {task['id']}"},
    ]
    built = SimpleNamespace(
        menu_sha256=MENU_SHA if scope == "full" else "e" * 64,
        schemas=[
            {"type": "function", "function": {"name": f"tool_{i}"}} for i in range(33)
        ]
        if scope == "full"
        else [{"type": "function", "function": {"name": "gold"}}],
    )
    return messages, built


def _artifacts(actions: list[str]):
    tasks = [_task(index + 1, action) for index, action in enumerate(actions)]
    scoped_tasks = deepcopy(tasks)
    for task in scoped_tasks:
        task["tool_scope"] = "scoped"
    labels = []
    no_tool_rows = []
    scoped_rows = []
    for task, scoped_task, action in zip(tasks, scoped_tasks, actions, strict=True):
        label = {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "category": task["category"],
            "gold_env_name": task["gold_env_name"],
            "tool_necessary": 0 if action == "NONE" else 1,
            "no_tool_correct": 1 if action == "NONE" else 0,
            "gold_action": action,
            "split": "train",
            "seed": 0,
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": "full",
            "final_response": f"\\boxed{{{task['id']}}}",
        }
        labels.append(label)
        no_tool_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": task["id"],
                "difficulty": task["difficulty"],
                "category": task["category"],
                "gold_env_name": task["gold_env_name"],
                "seed": 0,
                "tool_scope": "full",
                "prompt_mode": "hard_no_tool",
                "reasoning_mode": "no_reasoning",
                "run_id": "labels_seed_0",
                "final_correct": action == "NONE",
                "total_tool_calls": 0,
                "routed_tool_events": [],
                "invalid_tool_calls": 0,
                "episode_done": True,
                "termination_reason": "boxed_answer",
                "final_response": f"\\boxed{{{task['id']}}}",
            }
        )
        scoped_prefix, _ = _initial_builder(
            scoped_task,
            system_prompt="SYSTEM",
            setting=SimpleNamespace(tool_scope="scoped"),
        )
        scoped_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": task["id"],
                "difficulty": task["difficulty"],
                "category": task["category"],
                "gold_env_name": task["gold_env_name"],
                "seed": 0,
                "tool_scope": "scoped",
                "prompt_mode": "current",
                "reasoning_mode": "no_reasoning",
                "final_correct": True,
                "gold_action": action,
                "first_tool_category": action if action != "NONE" else task["category"],
                "total_tool_calls": 1,
                "invalid_tool_calls": 0,
                "tool_call_categories": [task["category"]],
                "routed_tool_events": [{"category": task["category"]}],
                "episode_done": True,
                "termination_reason": "boxed_answer",
                "final_response": f"\\boxed{{{task['id']}}}",
                "output": [
                    *deepcopy(scoped_prefix),
                    {
                        "role": "assistant",
                        "content": '<tool_call>\n{"name":"gold","arguments":{}}\n</tool_call>',
                    },
                    {
                        "role": "user",
                        "content": '<tool_response>\n{"success":true}\n</tool_response>',
                    },
                    {"role": "assistant", "content": f"\\boxed{{{task['id']}}}"},
                ],
            }
        )
    return (
        tasks,
        scoped_tasks,
        {
            "schema_version": SCHEMA_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "model": "test-model",
            "split": "train",
            "seed": 0,
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": "full",
            "n": len(tasks),
            "rows": labels,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "split": "train",
            "seed": 0,
            "setting": "hard_no_tool_no_reasoning_fulltools",
            "rows": no_tool_rows,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "config": {
                "model": "test-model",
                "data_sha256": SOURCE_PROVENANCE["scoped_data"]["sha256"],
                "labels_sha256": SOURCE_PROVENANCE["labels"]["sha256"],
                "runtime_provenance_sha256": RUNTIME_PROVENANCE["sha256"],
                "project_git_commit": RUNTIME_PROVENANCE["project_git_commit"],
                "full_menu_sha256": MENU_SHA,
                "task_ids_sha256": canonical_json_sha256(
                    [task["id"] for task in tasks]
                ),
                "smoke": False,
                "tool_scope": "scoped",
                "prompt_mode": "current",
                "reasoning_mode": "no_reasoning",
                "record_mode": "full",
                "seeds": [0],
            },
            "runs": [{"run_id": "run_0_seed_0", "seed": 0, "rows": scoped_rows}],
        },
    )


def _build(actions: list[str]):
    tasks, scoped_tasks, labels, no_tool, scoped = _artifacts(actions)
    return build_sft_trajectories(
        tasks,
        scoped_tasks,
        labels,
        no_tool,
        scoped,
        system_prompt="SYSTEM",
        source_provenance=SOURCE_PROVENANCE,
        expected_model_slug="test-model",
        expected_full_menu_sha256=MENU_SHA,
        runtime_provenance=RUNTIME_PROVENANCE,
        expected_task_count=len(tasks),
        initial_builder=_initial_builder,
    )


def test_scoped_prompt_is_verified_but_never_copied_into_sft() -> None:
    result = _build(["NONE", "A", "B", "C"])
    assert len(result.records) == 4
    assert result.manifest["decisions"]["retained"] == 4
    for record in result.records:
        contents = [message["content"] for message in record["messages"]]
        assert any(content.startswith("FULL CURRENT") for content in contents)
        assert not any("SCOPED CURRENT" in content for content in contents)
        assert record["full_menu_sha256"] == MENU_SHA
        assert len(record["input_sha256"]) == 64
    none = next(row for row in result.records if row["gold_action"] == "NONE")
    assert [message["role"] for message in none["messages"][-1:]] == ["assistant"]


def test_drop_reason_is_stratified_and_other_action_example_keeps_build_valid() -> None:
    tasks, scoped_tasks, labels, no_tool, scoped = _artifacts(
        ["NONE", "A", "A", "B", "C"]
    )
    scoped["runs"][0]["rows"][1]["first_tool_category"] = "B"
    result = build_sft_trajectories(
        tasks,
        scoped_tasks,
        labels,
        no_tool,
        scoped,
        system_prompt="SYSTEM",
        source_provenance=SOURCE_PROVENANCE,
        expected_model_slug="test-model",
        expected_full_menu_sha256=MENU_SHA,
        runtime_provenance=RUNTIME_PROVENANCE,
        expected_task_count=5,
        initial_builder=_initial_builder,
    )
    decisions = result.manifest["decisions"]
    assert decisions["dropped"] == 1
    assert decisions["reason_counts"]["dropped_tool_first_category_mismatch"] == 1
    assert decisions["by_action"]["A"]["dropped"] == 1
    env = tasks[1]["gold_env_name"]
    assert decisions["by_environment"][env]["dropped"] == 1
    difficulty = tasks[1]["difficulty"]
    assert decisions["by_difficulty"][difficulty]["dropped"] == 1


def test_missing_retained_action_fails_closed() -> None:
    tasks, scoped_tasks, labels, no_tool, scoped = _artifacts(["NONE", "A", "B", "C"])
    scoped["runs"][0]["rows"][2]["output"][0]["content"] = "tampered"
    with pytest.raises(ValueError, match="retain every action"):
        build_sft_trajectories(
            tasks,
            scoped_tasks,
            labels,
            no_tool,
            scoped,
            system_prompt="SYSTEM",
            source_provenance=SOURCE_PROVENANCE,
            expected_model_slug="test-model",
            expected_full_menu_sha256=MENU_SHA,
            runtime_provenance=RUNTIME_PROVENANCE,
            expected_task_count=4,
            initial_builder=_initial_builder,
        )


def test_scoped_source_must_share_stage7_runtime_receipt() -> None:
    tasks, scoped_tasks, labels, no_tool, scoped = _artifacts(["NONE", "A", "B", "C"])
    scoped["config"]["runtime_provenance_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime_provenance_sha256"):
        build_sft_trajectories(
            tasks,
            scoped_tasks,
            labels,
            no_tool,
            scoped,
            system_prompt="SYSTEM",
            source_provenance=SOURCE_PROVENANCE,
            expected_model_slug="test-model",
            expected_full_menu_sha256=MENU_SHA,
            runtime_provenance=RUNTIME_PROVENANCE,
            expected_task_count=4,
            initial_builder=_initial_builder,
        )


def test_atomic_jsonl_receipt_is_validated_by_training_loader(tmp_path: Path) -> None:
    result = _build(["NONE", "A", "B", "C"])
    output = tmp_path / "train_action_trajectories.jsonl"
    manifest_path = tmp_path / "train_action_trajectories.manifest.json"
    manifest = write_sft_artifacts(output, manifest_path, result)
    records, loaded_manifest = load_sft_records(
        output, manifest_path, expected_full_menu_sha256=MENU_SHA
    )
    assert len(records) == 4
    assert loaded_manifest == manifest
    with pytest.raises(FileExistsError):
        write_sft_artifacts(output, manifest_path, result)
    output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size"):
        load_sft_records(output, manifest_path, expected_full_menu_sha256=MENU_SHA)
