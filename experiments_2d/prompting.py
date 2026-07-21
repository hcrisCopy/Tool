"""Prompt variants used by labeling, hidden extraction, and final evaluation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .constants import CATEGORY_NAMES
from .upstream import build_environment_tools


SYSTEM_PROMPT = (
    "You can use tools when helpful. "
    "If you call a tool, emit exactly one tool call wrapped in "
    "<tool_call>...</tool_call>, with exactly one JSON object inside: "
    '{"name": "tool_name", "arguments": {...}}. '
    "Do not call multiple tools at once. "
    "Tool arguments must strictly match each tool schema; do not invent wrapper fields. "
    "When you are done, provide the final answer in LaTeX boxed format: \\boxed{...}."
)


def current_user_message(instruction: str) -> str:
    return (
        instruction
        + "\n\nResponse policy (required every turn):\n"
        + "1) You can choose to use a tool or not in this task.\n"
        + "2) Provide final answer in \\boxed{...} if you think the task is complete."
    )


def no_tool_user_message(instruction: str) -> str:
    return (
        instruction
        + "\n\nResponse policy (required every turn):\n"
        + "1) Do not use any tools in this task.\n"
        + "2) Provide final answer in \\boxed{...} if you think the task is complete."
    )


def all_type_tools() -> list[dict[str, Any]]:
    """Return a fixed three-tool P_all menu shared by every sample."""

    tools: list[dict[str, Any]] = []
    for category in ("A", "B", "C"):
        semantic_name = CATEGORY_NAMES[category]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"type_{category.lower()}_{semantic_name.replace('-', '_')}",
                    "description": (
                        f"Route a type-{category} ({semantic_name}) task to the "
                        "appropriate concrete environment operation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "environment": {"type": "string"},
                            "operation": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["environment", "operation", "arguments"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def tools_for_variant(task: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    if variant == "P_env":
        return build_environment_tools(task)
    if variant == "P_all":
        return deepcopy(all_type_tools())
    if variant == "P_no_schema":
        return []
    raise ValueError(f"Unsupported prompt variant: {variant}")


def initial_messages(task: dict[str, Any], *, no_tool: bool) -> list[dict[str, str]]:
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Task {task.get('id')} has an invalid instruction")
    user = no_tool_user_message(instruction) if no_tool else current_user_message(instruction)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer returned an empty chat-template rendering")
    return rendered


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(token_ids, list) or not token_ids:
        raise TypeError("Tokenizer did not return a non-empty token-id list")
    if not all(isinstance(token_id, int) for token_id in token_ids):
        raise TypeError("Chat-template token ids must all be integers")
    return token_ids


def binary_prefill(tool_needed: bool) -> str:
    if tool_needed:
        return "I need to use a tool for this question."
    return "I can solve this directly without using a tool."


def binary_type_prefill(tool_needed: bool, category: str) -> str:
    if category not in CATEGORY_NAMES:
        raise ValueError(f"Unsupported category: {category}")
    name = CATEGORY_NAMES[category]
    if tool_needed:
        return f"I need to use a {name} tool for this question."
    return f"I can solve this directly without using a tool. This is a {name} task."


def structured_prefill(tool_needed: bool, category: str) -> str:
    if category not in CATEGORY_NAMES:
        raise ValueError(f"Unsupported category: {category}")
    decision = "yes" if tool_needed else "no"
    return f"Decision: tool_needed={decision}; task_type={category}."
