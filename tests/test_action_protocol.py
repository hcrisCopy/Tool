from __future__ import annotations

from copy import deepcopy

import pytest

from when2tool_action.constants import (
    ACTIONS,
    CATEGORY_TO_ENVS,
    ENV_TO_CATEGORY,
    EXPECTED_ENV_COUNT,
    EXPECTED_TOOL_COUNT,
)
from when2tool_action.labels import build_label_rows
from when2tool_action.runtime import (
    EvaluationSetting,
    attach_gold_actions,
    initial_messages_and_tools,
)
from when2tool_action.safety import (
    normalize_code,
    trusted_code_from_tasks,
    validate_tool_arguments,
)
from when2tool_action.upstream import build_environments, frozen_full_menu


def _task(task_id: int = 1, env: str = "CalculatorEnv") -> dict:
    category = ENV_TO_CATEGORY[env]
    tools = {
        "CalculatorEnv": ["evaluate_expression", "get_last_result", "clear_last_result"],
        "CodeExecutorEnv": ["run_code"],
    }[env]
    instruction = (
        "Run this code:\n```python\nprint(2 + 2)\n```"
        if env == "CodeExecutorEnv"
        else "What is 2+2?"
    )
    return {
        "id": task_id,
        "difficulty": "easy",
        "multi_step": False,
        "instruction": instruction,
        "environments": [{"name": env, "tools": tools, "parameters": {}}],
        "expected": {"answer": "4"},
        "tags": [],
        "category": category,
        "category_name": "test",
        "gold_env_name": env,
        "gold_tools": tools,
        "tool_scope": "full",
    }


def test_category_map_and_original_full_menu_are_frozen() -> None:
    assert ACTIONS == ("NONE", "A", "B", "C")
    assert len(ENV_TO_CATEGORY) == EXPECTED_ENV_COUNT
    assert all(len(envs) == 5 for envs in CATEGORY_TO_ENVS.values())
    schemas, routes, digest = frozen_full_menu()
    names = [schema["function"]["name"] for schema in schemas]
    assert len(schemas) == EXPECTED_TOOL_COUNT
    assert len(names) == len(set(names))
    assert len(routes) == EXPECTED_TOOL_COUNT
    assert all("__" not in name for name in names)
    assert len(digest) == 64


def test_full_builder_keeps_gold_task_metadata_and_menu_order() -> None:
    task = _task()
    before = deepcopy(task)
    first = build_environments(task, "full")
    second = build_environments(task, "full")
    assert task == before
    assert first.schemas == second.schemas
    assert first.menu_sha256 == second.menu_sha256
    assert len(first.envs) == EXPECTED_ENV_COUNT
    assert len(first.schemas) == EXPECTED_TOOL_COUNT
    assert first.route_map["evaluate_expression"].category == "A"
    assert first.route_map["search_corpus"].category == "B"
    assert first.route_map["run_code"].category == "C"


def test_full_prompt_contract_does_not_depend_on_gold_environment() -> None:
    setting = EvaluationSetting(
        name="current_no_reasoning_fulltools",
        tool_scope="full",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="lite",
    )
    for env in ("CalculatorEnv", "CodeExecutorEnv"):
        messages, built = initial_messages_and_tools(
            _task(env=env), system_prompt="SYSTEM", setting=setting
        )
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "SYSTEM"}
        assert messages[1]["role"] == "system"
        assert messages[1]["content"].startswith(
            "ListManipulation format contract:"
        )
        assert len(built.schemas) == EXPECTED_TOOL_COUNT


def test_scoped_prompt_only_adds_contract_when_list_tools_are_exposed() -> None:
    setting = EvaluationSetting(
        name="current_no_reasoning_scoped",
        tool_scope="scoped",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="lite",
    )
    messages, _ = initial_messages_and_tools(
        _task(env="CalculatorEnv"), system_prompt="SYSTEM", setting=setting
    )
    assert len(messages) == 2


def test_hard_no_tool_labels_and_gold_attachment() -> None:
    task = _task()
    evaluated = [
        {
            **task,
            "total_tool_calls": 0,
            "routed_tool_events": [],
            "final_correct": False,
            "final_response": "\\boxed{5}",
        }
    ]
    labels = build_label_rows(
        evaluated, split="test", seed=0, tool_scope="full"
    )
    assert labels[0]["tool_necessary"] == 1
    assert labels[0]["gold_action"] == "A"
    attached = attach_gold_actions([task], labels)
    assert attached[0]["gold_action"] == "A"
    bad = deepcopy(labels)
    bad[0]["gold_action"] = "NONE"
    with pytest.raises(ValueError, match="inconsistent gold action"):
        attach_gold_actions([task], bad)


def test_safety_policy_extracts_only_exact_fenced_code_and_caps_inputs() -> None:
    task = _task(env="CodeExecutorEnv")
    trusted = trusted_code_from_tasks([task])
    assert trusted == frozenset({"print(2 + 2)"})
    assert normalize_code("\r\nprint(2 + 2)\r\n") in trusted
    assert validate_tool_arguments("factorial", {"n": 1000}) == (True, None)
    valid, reason = validate_tool_arguments("factorial", {"n": 1001})
    assert not valid and "1000" in str(reason)
    valid, reason = validate_tool_arguments(
        "matrix_determinant", {"matrix": [[1] * 11] * 11}
    )
    assert not valid and "10" in str(reason)
    valid, reason = validate_tool_arguments(
        "evaluate_expression", {"expression": "2 ** 10001"}
    )
    assert not valid and "10000" in str(reason)
