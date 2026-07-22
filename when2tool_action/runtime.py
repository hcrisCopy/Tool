"""Full/scoped When2Tool evaluator with explicit routed-event accounting."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .constants import ACTIONS, ENV_TO_CATEGORY, SCHEMA_VERSION
from .safety import (
    compact_json,
    run_trusted_code,
    trusted_code_from_tasks,
    validate_tool_arguments,
    wall_clock_timeout,
)
from .upstream import BuiltEnvironments, build_environments, load_runtime


@dataclass(frozen=True)
class EvaluationSetting:
    name: str
    tool_scope: str
    prompt_mode: str
    require_reasoning: bool
    record_mode: str = "lite"


def _count_tokens(state: dict[str, Any], text: str) -> int:
    if not text:
        return 0
    return len(state["tokenizer"].encode(text, add_special_tokens=False))


def _initial_state(
    task: dict[str, Any], agent: Any, setting: EvaluationSetting
) -> dict[str, Any]:
    messages, built = initial_messages_and_tools(
        task,
        system_prompt=agent.system_prompt,
        setting=setting,
    )
    utils, _, _ = load_runtime()
    tool_format = utils.detect_tool_format("qwen3-4b-instruct")
    return {
        "task": deepcopy(task),
        "envs": built.envs,
        "tools": built.schemas,
        "route_map": built.route_map,
        "menu_sha256": built.menu_sha256,
        "messages": messages,
        "rounds": 0,
        "done": False,
        "trace": [],
        "routed_tool_events": [],
        "record_mode": setting.record_mode,
        "final_response": "",
        "prompt_mode": setting.prompt_mode,
        "require_reasoning": setting.require_reasoning,
        "tool_calls_used": 0,
        "tool_format": tool_format,
        "tokenizer": agent.tokenizer,
        "generation_tokens": 0,
        "prefill_tokens": 0,
    }


def initial_messages_and_tools(
    task: dict[str, Any],
    *,
    system_prompt: str,
    setting: EvaluationSetting,
) -> tuple[list[dict[str, str]], BuiltEnvironments]:
    """Return the exact first-round prompt contract used for eval and probes."""

    utils, _, _ = load_runtime()
    built = build_environments(task, setting.tool_scope)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # The ListManipulation contract is a property of the exposed tool menu, not
    # of the hidden gold environment.  In full-menu runs those tools are
    # available for every task, so every task must receive the same contract;
    # conditioning this message on gold_env_name would leak the target.
    exposes_list_tools = (
        setting.tool_scope == "full"
        or task["gold_env_name"] == "ListManipulationEnv"
    )
    if exposes_list_tools:
        messages.append(
            {
                "role": "system",
                "content": (
                    "ListManipulation format contract:\n"
                    "1) There is no set_list tool. For every list-op call, you must provide values=<current list> explicitly.\n"
                    "2) For tool arguments, use plain lists [a,b,c] (1D) or [[...],[...]] (2D). Never use objects like {\"values\": [...]}.\n"
                    "3) At each step, parse current_list from the latest tool output, then call exactly one operation tool (append/remove/insert/sort/reverse).\n"
                    "4) Use one operation per tool call; do not batch multiple operations in one call.\n"
                    "5) For final answer, output exactly one plain list literal wrapped in box, e.g. \\boxed{[1, 2, 3]} or \\boxed{[[1, 2], [3, 4]]}; do NOT format it as LaTeX array/matrix.\n"
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": utils.build_user_message(
                task["instruction"],
                setting.prompt_mode,
                require_reasoning=setting.require_reasoning,
            ),
        }
    )
    return messages, built


def _find_env(built_state: dict[str, Any], tool_name: str) -> tuple[Any, set[str]] | None:
    for env, allowed in built_state["envs"]:
        if tool_name in allowed and env.has_tool(tool_name):
            return env, allowed
    return None


def _route_and_record(
    state: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    trusted_code: frozenset[str],
) -> dict[str, Any]:
    info = state["route_map"].get(tool_name)
    recognized = info is not None
    arguments_valid, validation_error = validate_tool_arguments(tool_name, arguments)
    event = {
        "round": state["rounds"],
        "tool_name": tool_name,
        "arguments": deepcopy(arguments),
        "recognized": recognized,
        "environment": info.environment if info else None,
        "category": info.category if info else "INVALID",
        "arguments_valid": bool(arguments_valid),
        "result_success": None,
        "result": None,
        "routed": True,
    }
    # Persist the routing decision before execution.  This keeps the behavioral
    # choice auditable even if a guarded tool later times out.
    state["routed_tool_events"].append(event)
    result: dict[str, Any]
    if not recognized:
        result = {
            "success": False,
            "message": f"Tool {tool_name} not available in this task.",
        }
    elif not arguments_valid:
        result = {
            "success": False,
            "message": f"[SAFETY_REJECTED] {validation_error}",
        }
    elif tool_name == "run_code":
        eligible_code = (
            trusted_code_from_tasks([state["task"]])
            if state["task"]["gold_env_name"] == "CodeExecutorEnv"
            else frozenset()
        )
        if not eligible_code <= trusted_code:
            raise AssertionError("Per-task code allowlist is outside evaluated corpus")
        result = run_trusted_code(arguments.get("code"), eligible_code)
    else:
        target = _find_env(state, tool_name)
        if target is None:
            raise AssertionError(f"Recognized tool {tool_name} has no environment")
        env, _ = target
        # Capture schema coercion validity separately from our resource policy.
        description = env.get_tool_descs([tool_name])[0]
        _coerced, coercion_error = env._coerce_args(description, arguments)
        if coercion_error is not None:
            arguments_valid = False
        try:
            with wall_clock_timeout(5):
                result = env.call_tool(tool_name, arguments)
        except TimeoutError as error:
            result = {"success": False, "message": f"[TOOL_RUNTIME_TIMEOUT] {error}"}
    event["arguments_valid"] = bool(arguments_valid)
    event["result_success"] = bool(result.get("success"))
    event["result"] = compact_json(deepcopy(result))
    return result


def _trace_item(state: dict[str, Any], out: dict[str, Any]) -> dict[str, Any] | None:
    if state["record_mode"] == "off":
        return None
    prompt = out.get("prompt_text", "")
    item: dict[str, Any] = {
        "round": state["rounds"] + 1,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": _count_tokens(state, prompt),
        "model_raw_output": out.get("raw_text", ""),
        "model_finish_reason": out.get("finish_reason", "unknown"),
        "parsed_output": {
            "type": out.get("type", "unknown"),
            "tool_name": out.get("tool_name"),
            "arguments": deepcopy(out.get("arguments", {}))
            if out.get("type") == "tool"
            else None,
            "content": out.get("content", "")
            if out.get("type") == "content"
            else None,
        },
        "attempted_tool_parse_failure": bool(
            out.get("type") != "tool"
            and (
                "<tool_call>" in str(out.get("raw_text", ""))
                or "\"name\"" in str(out.get("raw_text", ""))
            )
        ),
    }
    if state["record_mode"] == "full":
        item["prompt_text"] = prompt
        item["tools"] = deepcopy(state["tools"])
    return item


def _step(
    state: dict[str, Any],
    out: dict[str, Any],
    trace_item: dict[str, Any] | None,
    trusted_code: frozenset[str],
) -> None:
    utils, _, _ = load_runtime()
    original_route = utils.route_tool_call

    def route(_envs: Any, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _route_and_record(state, tool_name, arguments, trusted_code)

    before_events = len(state["routed_tool_events"])
    before_calls = int(state["tool_calls_used"])
    utils.route_tool_call = route
    try:
        utils.step_state(state, out, trace_item=trace_item)
    finally:
        utils.route_tool_call = original_route
    event_delta = len(state["routed_tool_events"]) - before_events
    call_delta = int(state["tool_calls_used"]) - before_calls
    if event_delta != call_delta or event_delta not in {0, 1}:
        raise AssertionError(
            f"Routed event/tool count delta mismatch: {event_delta}/{call_delta}"
        )


def _finalize(
    state: dict[str, Any], setting: EvaluationSetting, run_id: str, seed: int
) -> dict[str, Any]:
    utils, _, _ = load_runtime()
    result = utils.finalize_state(state)
    events = deepcopy(state["routed_tool_events"])
    if len(events) != int(result["tool_calls"]):
        raise AssertionError(
            f"Task {result['id']} events={len(events)} tool_calls={result['tool_calls']}"
        )
    categories = [event["category"] for event in events]
    unique_categories = list(dict.fromkeys(categories))
    pred_action = "NONE" if not categories else categories[0]
    if pred_action not in {*ACTIONS, "INVALID"}:
        raise AssertionError(f"Unexpected predicted action {pred_action}")
    _raw, _boxed, _cleaned, final_correct = utils.item_final_eval(result)
    gold_tools = set(result["gold_tools"])
    first = events[0] if events else None
    gold_action = result.get("gold_action")
    error_type: str | None = None
    if gold_action in ACTIONS:
        if "INVALID" in categories:
            error_type = "invalid_tool"
        elif gold_action == "NONE" and pred_action != "NONE":
            error_type = "over_call"
        elif gold_action != "NONE" and pred_action == "NONE":
            error_type = "under_call"
        elif gold_action != "NONE" and pred_action != gold_action:
            error_type = "wrong_category"
        elif not final_correct and pred_action == "NONE":
            error_type = "direct_answer_wrong"
        elif not final_correct:
            error_type = "correct_category_wrong_answer"
        else:
            error_type = "success"
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "setting": setting.name,
            "tool_scope": setting.tool_scope,
            "run_id": run_id,
            "seed": seed,
            "menu_sha256": state["menu_sha256"],
            "routed_tool_events": events,
            "total_tool_calls": len(events),
            "first_tool_name": first["tool_name"] if first else None,
            "first_tool_category": first["category"] if first else None,
            "first_tool_environment": first["environment"] if first else None,
            "first_env_correct": bool(first and first["environment"] == result["gold_env_name"]),
            "exact_tool_allowed": bool(first and first["tool_name"] in gold_tools),
            "first_arguments_valid": bool(first and first["arguments_valid"]),
            "pred_action": pred_action,
            "tool_call_categories": categories,
            "unique_tool_call_categories": unique_categories,
            "n_tool_call_categories": len(unique_categories),
            "mixed_category_calls": len(set(categories)) >= 2,
            "invalid_tool_calls": sum(category == "INVALID" for category in categories),
            "final_correct": bool(final_correct),
            "error_type": error_type,
            "episode_done": bool(state["done"]),
            "termination_reason": "boxed_answer" if state["done"] else "max_rounds",
            "tool_parse_failures": sum(
                bool(item.get("attempted_tool_parse_failure"))
                for item in state.get("trace", [])
            ),
        }
    )
    return result


def evaluate(
    tasks: list[dict[str, Any]],
    agent: Any,
    setting: EvaluationSetting,
    *,
    seed: int,
    run_id: str,
    max_rounds: int,
    max_model_len: int,
    prefills: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    if not tasks:
        raise ValueError("No evaluation tasks")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation task IDs")
    if prefills is not None and set(prefills) != set(ids):
        raise ValueError(
            f"Prefill IDs differ from tasks: missing={sorted(set(ids)-set(prefills))[:10]}, "
            f"extra={sorted(set(prefills)-set(ids))[:10]}"
        )
    trusted_code = trusted_code_from_tasks(tasks)
    states = [_initial_state(task, agent, setting) for task in tasks]
    menu_hashes = {state["menu_sha256"] for state in states}
    if setting.tool_scope == "full" and len(menu_hashes) != 1:
        raise AssertionError("Full menu is not identical across tasks")

    for round_number in range(1, max_rounds + 1):
        active = [
            index
            for index, state in enumerate(states)
            if not state["done"] and state["rounds"] < max_rounds
        ]
        if not active:
            break
        messages_batch = [states[index]["messages"] for index in active]
        tools_batch = [states[index]["tools"] for index in active]
        round_prefills: list[str | None] | None = None
        if round_number == 1 and prefills is not None:
            round_prefills = [prefills[states[index]["task"]["id"]] for index in active]
            if any(not isinstance(prefill, str) or not prefill for prefill in round_prefills):
                raise ValueError("Every prefill must be a non-empty string")
            for index in active:
                states[index]["has_prefill"] = True
        for batch_index, state_index in enumerate(active):
            state = states[state_index]
            kwargs = {
                "tools": state["tools"],
                "add_generation_prompt": True,
                "tokenize": True,
            }
            try:
                prompt_ids = agent.tokenizer.apply_chat_template(
                    state["messages"], enable_thinking=False, **kwargs
                )
            except TypeError:
                prompt_ids = agent.tokenizer.apply_chat_template(state["messages"], **kwargs)
            prefill_tokens = (
                _count_tokens(state, round_prefills[batch_index])
                if round_prefills is not None
                else 0
            )
            if len(prompt_ids) + prefill_tokens > max_model_len - 128:
                raise ValueError(
                    f"Task {state['task']['id']} prompt exceeds context budget"
                )
        outs = agent.generate_batch(
            messages_batch, tools_batch, prefills=round_prefills
        )
        if len(outs) != len(active):
            raise RuntimeError("Model output count differs from active task count")
        for state_index, out in zip(active, outs):
            state = states[state_index]
            _step(state, out, _trace_item(state, out), trusted_code)
        done = sum(state["done"] for state in states)
        print(
            f"[{setting.name} seed={seed}] round {round_number}: "
            f"active={len(active)} done={done}/{len(states)}",
            flush=True,
        )
    results = [
        _finalize(state, setting, run_id, seed)
        for state in tqdm(states, desc=f"finalize {setting.name}", unit="task")
    ]
    if [row["id"] for row in results] != ids:
        raise AssertionError("Evaluation changed task order")
    return results


def attach_gold_actions(
    tasks: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = {row["id"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("Duplicate label IDs")
    if set(labels) != {task["id"] for task in tasks}:
        raise ValueError("Label/task ID sets differ")
    output: list[dict[str, Any]] = []
    for task in tasks:
        label = labels[task["id"]]
        expected = task["category"] if label["tool_necessary"] else "NONE"
        if label.get("gold_action") != expected:
            raise ValueError(f"Task {task['id']} has inconsistent gold action")
        if label.get("category") != task["category"]:
            raise ValueError(f"Task {task['id']} label category mismatch")
        output.append(dict(deepcopy(task), gold_action=expected))
    return output
